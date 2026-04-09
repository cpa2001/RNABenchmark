import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import random
import sklearn
import scipy
import transformers

import numpy as np
from torch.utils.data import Dataset


os.environ["WANDB_DISABLED"] = "true"


from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.rnalm.modeling_rnalm import RnaLmForSequenceClassification
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForSequenceClassification
from model.rnabert.modeling_rnabert import RnaBertForSequenceClassification
from model.rnamsm.modeling_rnamsm import RnaMsmForSequenceClassification
from model.splicebert.modeling_splicebert import SpliceBertForSequenceClassification
from model.utrbert.modeling_utrbert import UtrBertForSequenceClassification
from model.utrlm.modeling_utrlm import UtrLmForSequenceClassification
from model.ecorna.modeling_ecorna import EcoRNAForSequenceClassification
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

# Add ecorna repo to path for EcoRNA imports
ecorna_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(parent_dir)))
if ecorna_repo_root not in sys.path:
    sys.path.insert(0, ecorna_repo_root)
early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    metric_for_best_model: str = field(default="accuracy")
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    ecorna_pooling_strategy: str = field(default="weighted_layer_content")
    ecorna_pooling_cells: str = field(default="")
    ecorna_num_loops: int = field(default=-1)
    freeze_backbone: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


"""
Transform a rna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from rna sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each rna sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each rna sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0].upper().replace("U", "T") for d in data]
            labels = [int(d[1]) for d in data]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        text = texts[0]
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        # ensure tokenier
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
        self.labels = labels
        self.num_labels = len(set(labels))
        self.texts = texts

    def __len__(self):
       return len(self.texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.texts[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        seqs, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        output = self.tokenizer(seqs, padding='longest', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def set_module_trainable(module, trainable):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = trainable


def get_trainable_parameter_names(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_ecorna_runtime_diagnostics(model):
    base_model = unwrap_model(model)
    if hasattr(base_model, "get_runtime_diagnostics"):
        return base_model.get_runtime_diagnostics()
    if hasattr(base_model, "get_loop_layer_mix_stats"):
        return base_model.get_loop_layer_mix_stats()
    return {}


def reset_ecorna_runtime_diagnostics(model):
    base_model = unwrap_model(model)
    if hasattr(base_model, "reset_runtime_diagnostics"):
        base_model.reset_runtime_diagnostics()


def get_frozen_readout_component_names(model, training_args):
    names = [name for name, _ in get_frozen_readout_modules(model, training_args)]
    if (
        training_args.model_type == 'ecorna'
        and training_args.ecorna_pooling_strategy in ("layer_weighted", "loop_layer_scalar_mix_content")
        and getattr(model, "loop_layer_mix_logits", None) is not None
    ):
        names.append("pooler.loop_layer_mix_logits")
    return names


def log_and_validate_ecorna_special_tokens(tokenizer, model, training_args):
    base_model = unwrap_model(model)
    config = base_model.config
    token_attrs = ("cls_token_id", "sep_token_id", "eos_token_id", "pad_token_id")
    tokenizer_truth = {attr: getattr(tokenizer, attr, None) for attr in token_attrs}
    config_truth = {attr: getattr(config, attr, None) for attr in token_attrs}

    mismatches = [
        attr for attr in token_attrs
        if tokenizer_truth[attr] != config_truth[attr]
    ]
    if mismatches:
        raise ValueError(
            f"EcoRNA tokenizer/config special token mismatch for {mismatches}: "
            f"tokenizer={tokenizer_truth}, config={config_truth}"
        )

    sample_sequence = "AUGC"
    sample_encoding = tokenizer(sample_sequence, add_special_tokens=True)
    sample_ids = sample_encoding["input_ids"]
    sample_tokens = tokenizer.convert_ids_to_tokens(sample_ids)
    expected_cls = tokenizer_truth["cls_token_id"]
    expected_sep = tokenizer_truth["sep_token_id"]
    special_ids = {
        token_id
        for token_id in tokenizer_truth.values()
        if token_id is not None
    }

    if not sample_ids:
        raise ValueError("EcoRNA tokenizer produced an empty encoding for the startup sample.")
    if expected_cls is None or sample_ids[0] != expected_cls:
        raise ValueError(
            f"EcoRNA startup tokenization does not begin with CLS: ids={sample_ids}, tokens={sample_tokens}"
        )
    if expected_sep is None or sample_ids[-1] != expected_sep:
        raise ValueError(
            f"EcoRNA startup tokenization does not end with SEP: ids={sample_ids}, tokens={sample_tokens}"
        )
    if len(sample_ids) < 3:
        raise ValueError(
            f"EcoRNA startup tokenization is too short to validate layout: ids={sample_ids}, tokens={sample_tokens}"
        )

    content_ids = sample_ids[1:-1]
    leaked_special_ids = [token_id for token_id in content_ids if token_id in special_ids]
    if leaked_special_ids:
        raise ValueError(
            "EcoRNA startup tokenization violates the expected [CLS] + content + [SEP] layout: "
            f"ids={sample_ids}, tokens={sample_tokens}, leaked_special_ids={leaked_special_ids}"
        )

    if training_args.local_rank in [-1, 0]:
        print("EcoRNA tokenizer/config special token truth:")
        print(f"  tokenizer={tokenizer_truth}")
        print(f"  config={config_truth}")
        print("EcoRNA startup tokenization check:")
        print(f"  sample_sequence={sample_sequence}")
        print(f"  sample_ids={sample_ids}")
        print(f"  sample_tokens={sample_tokens}")


def get_frozen_readout_modules(model, training_args):
    if training_args.model_type == 'rna-fm':
        return [("rnafm.pooler", model.rnafm.pooler), ("classifier", model.classifier)]

    if training_args.model_type != 'ecorna':
        raise ValueError(
            f"freeze_backbone is only supported for rna-fm and ecorna, got {training_args.model_type}"
        )

    strategy = training_args.ecorna_pooling_strategy
    modules = [("classifier", model.classifier)]

    if strategy in (
        "cls",
        "mean",
        "content_mean",
        "loop_mean_content",
        "loop_mean_cls",
        "cls_mean_concat",
        "fixed_cell_content",
        "fixed_cell_cls",
    ):
        pass
    elif strategy in (
        "layer_weighted",
        "loop_layer_scalar_mix_content",
        "loop_layer_attn_content",
        "weighted_layer_content",
        "weighted_cell_content",
    ):
        modules.insert(0, ("pooler", model.pooler))
    elif strategy == "cls_tanh":
        modules.insert(0, ("pooler", model.pooler))
    elif strategy == "cls_ln":
        modules.insert(0, ("cls_norm", model.cls_norm))
    else:
        raise ValueError(f"Unsupported pooling strategy for frozen readout: {strategy}")

    return modules


def maybe_freeze_backbone(model, training_args):
    if not training_args.freeze_backbone:
        return

    for param in model.parameters():
        param.requires_grad = False

    for _, module in get_frozen_readout_modules(model, training_args):
        set_module_trainable(module, True)

    if (
        training_args.model_type == 'ecorna'
        and training_args.ecorna_pooling_strategy in ("layer_weighted", "loop_layer_scalar_mix_content")
        and getattr(model, "loop_layer_mix_logits", None) is not None
    ):
        model.loop_layer_mix_logits.requires_grad = True

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type == 'rnalm':
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'ecorna':
        from ecorna import EcoRNATokenizer
        # EcoRNA uses single-char tokenizer, replace_u_with_t=True to match BEACON convention
        tokenizer = EcoRNATokenizer(
            replace_u_with_t=True,
            model_max_length=training_args.model_max_length,
        )
    elif training_args.model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        # Initialize tokenizer directly instead of from_pretrained to avoid compatibility issues
        tokenizer = OpenRnaLMTokenizer(
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )


    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    if 'mer' in training_args.token_type:
        data_args.kmer=int(training_args.token_type[0])
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # load model
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            print('Train from scratch')
            config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                token_type=training_args.token_type,
                problem_type="single_label_classification",
                attn_implementation=training_args.attn_implementation,
                )
            print(config)
            model =  RnaLmForSequenceClassification(
                config,
                )
        else:
            print('Loading rnalm model')
            print(train_dataset.num_labels)
            model =  RnaLmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                )
    elif training_args.model_type == 'rna-fm':      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaFmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'rnabert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'rnamsm':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaMsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )        
    elif 'splicebert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = SpliceBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )       
    elif 'utrbert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )  
    elif 'utr-lm' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrLmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )
    elif training_args.model_type == 'ecorna':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        num_loops = training_args.ecorna_num_loops if training_args.ecorna_num_loops > 0 else None
        model = EcoRNAForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=True,
            token_type=training_args.token_type,
            pooling_strategy=training_args.ecorna_pooling_strategy,
            pooling_cells=training_args.ecorna_pooling_cells,
            num_loops=num_loops,
        )     
    else:
        raise ValueError(f"Unsupported model_type: {training_args.model_type}")

    maybe_freeze_backbone(model, training_args)
    if training_args.model_type == 'ecorna':
        log_and_validate_ecorna_special_tokens(tokenizer, model, training_args)
    parameter_counts = get_parameter_number(model)
    if training_args.local_rank in [-1, 0]:
        print(f"Freeze backbone: {training_args.freeze_backbone}")
        print(f"Parameter counts: {parameter_counts}")
        if training_args.freeze_backbone:
            print(
                "Frozen readout modules:",
                get_frozen_readout_component_names(model, training_args),
            )
            print("Trainable parameters:")
            for name in get_trainable_parameter_names(model):
                print(f"  - {name}")
        if training_args.model_type == 'ecorna':
            print("EcoRNA initial diagnostics:")
            print(f"  {get_ecorna_runtime_diagnostics(model)}")

    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        if training_args.model_type == 'ecorna':
            reset_ecorna_runtime_diagnostics(trainer.model)
        results = trainer.evaluate(eval_dataset=test_dataset)
        if training_args.model_type == 'ecorna':
            runtime_diagnostics = get_ecorna_runtime_diagnostics(trainer.model)
            results.update(runtime_diagnostics)
            if training_args.local_rank in [-1, 0]:
                print("EcoRNA runtime diagnostics:")
                print(f"  {runtime_diagnostics}")
        print("on the test set:", results, "\n", results_path)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)
         




if __name__ == "__main__":
    train()
