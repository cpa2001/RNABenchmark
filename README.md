# BEACON: Benchmark for Comprehensive RNA Tasks and Language Models

This is the official codebase of the paper [BEACON: Benchmark for Comprehensive RNA Tasks and Language Models](https://arxiv.org/abs/2406.10391)

<p align="center">
    <img src="images/main.png" width="100%" height="100%">
</p>

## 🔥 Update
- [07/25]🔥 Updating models list and usage!
- [06/11]🔥 BEACON is coming! We release the [paper](https://arxiv.org/abs/2406.10391), [code](https://github.com/terry-r123/RNABenchmark), [data](https://drive.google.com/drive/folders/19ddrwI8ycvIxkgSV3gDo_VunLofYd4-6?usp=sharing), and [models](https://drive.google.com/drive/folders/1455JIOGV5X96CCgxCT-QgVu0xbXFz72X?usp=sharing) for BEACON!
- [02/15] Added EcoRNA inference/finetuning integration for BEACON tasks (opensource pipeline).
- [04/09] EcoRNA NoncodingRNAFamily release now defaults to the `weighted_layer_content` pooler in the opensource benchmark launcher.

## Prerequisites

### Installation
important libs:  torch==1.13.1+cu117, transformers==4.38.1


```bash
git clone https://github.com/terry-r123/RNABenchmark.git
cd RNABenchmark
conda create -n beacon python=3.8
pip install -r requirements.txt
```

For EcoRNA, use a modern PyTorch/Transformers stack. In practice, these are required:
- `torch>=2.2`
- `transformers>=4.40`
- `accelerate>=0.27`
- `tokenizers>=0.15`
- `safetensors>=0.4`
- `liger-kernel` (required if the checkpoint was trained with Liger MLP)
- `flash-attn` (recommended for speed; numerically equivalent attention output)

## 🔍 Tasks and Datasets

Datasets of RNA tasks can be found in [Google Drive](https://drive.google.com/drive/folders/19ddrwI8ycvIxkgSV3gDo_VunLofYd4-6?usp=sharing)

Model checkpoints of opensource RNA language models and BEACON-B can be found in [Google Drive](https://drive.google.com/drive/folders/1455JIOGV5X96CCgxCT-QgVu0xbXFz72X?usp=sharing)

### Data structure
```
RNABenchmark
├── checkpoint
│   └── opensource
|       ├── rna-fm
|       ├── rnabert
|       ├── rnamsm
|       ├── splicebert-human510
|       ├── splicebert-ms510
|       ├── splicebert-ms1024
|       ├── utr-lm-mrl    
|       ├── utr-lm-te-el    
|       ├── utrbert-3mer    
|       ├── utrbert-4mer  
|       ├── utrbert-5mer  
|       └── utrbert-6mer   
│   └── baseline
|       ├── BEACON-B
|       └── BEACON-B512
├── data
│    ├── ContactMap
│    ├── CRISPROffTarget
│    ├── CRISPROnTarget
│    ├── Degradation
│    ├── DistanceMap
│    ├── Isoform
│    ├── MeanRibosomeLoading
│    ├── Modification
│    ├── NoncodingRNAFamily
│    ├── ProgrammableRNASwitches
│    ├── Secondary_structure_prediction
│    ├── SpliceAI
│    └── StructuralScoreImputation
├── downstream
│   └── structure
├── model
|   |── rna-fm
|   ├── rnabert
|   ├── rnamsm
|   ├── splicebert
|   ├── utrlm      
|   ├── utrbert   
|   └── rnalm  
├── tokenizer
└── scripts
│    ├── BEACON-B
│    └── opensource
```


The full list of current task names are : 

- `Secondary_structure_prediction`
- `ContactMap`
- `DistanceMap`
- `StructuralScoreImputation`
- `SpliceAI`
- `Isoform`
- `NoncodingRNAFamily`
- `Modification`
- `MeanRibosomeLoading`
- `Degradation`
- `ProgrammableRNASwitches`
- `CRISPROnTarget`
- `CRISPROffTarget`


## 🔍Models 
<p align="center">
    <img src="images/exp1.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="images/exp2.png" width="100%" height="100%">
</p>

And the list of available embedders/models used for training on the tasks are : 

- `rna-fm`
- `rnabert`
- `rnamsm`
- `utr-lm-mrl`
- `utr-lm-te-el` 
- `splicebert-human510`
- `splicebert-ms510`
- `splicebert-ms1024`
- `utrbert-3mer`
- `utrbert-4mer`
- `utrbert-5mer`
- `utrbert-6mer`
### Model settings

| Models | name | token | pos | length| 
| --- | --- | --- | ---| --- |
|[RNA-FM](https://doi.org/10.48550/arXiv.2204.00300) | rna-fm | single  | ape| 1024| 
|[RNABERT](academic.oup.com/nargab/article/4/1/lqac012/6534363) | rnabert | single | ape| 440 | 
|[RNA-MSM](academic.oup.com/nar/article/52/1/e3/7369930)| rnamsm | single | ape | 1024
|[SpliceBERT-H510](academic.oup.com/bib/article/25/3/bbae163/7644137)| splicebert-human510 | single | ape | 510 |
|[SpliceBERT-MS510](academic.oup.com/bib/article/25/3/bbae163/7644137)| splicebert-ms510 | single | ape | 510 |
|[SpliceBERT-MS510](academic.oup.com/bib/article/25/3/bbae163/7644137)| splicebert-ms510 | single | ape | 1024 |
|[UTR-LM-MRL](https://www.nature.com/articles/s42256-024-00823-9) | utr-lm-mrl | single | rope | 1026 |
|[UTR-LM-TE&EL](https://www.nature.com/articles/s42256-024-00823-9)| utr-lm-te-el | single | rope | 1026 |
|[UTRBERT-3mer](https://doi.org/10.1101/2023.09.08.556883) | utrbert-3mer | 3mer |ape| 512 |
|[UTRBERT-4mer](https://doi.org/10.1101/2023.09.08.556883) | utrbert-4mer | 4mer |ape| 512 |
|[UTRBERT-5mer](https://doi.org/10.1101/2023.09.08.556883) | utrbert-5mer | 5mer |ape| 512 |
|[UTRBERT-6mer](https://doi.org/10.1101/2023.09.08.556883) | utrbert-6mer | 6mer |ape| 512 |
|[BEACON-B](https://arxiv.org/abs/2406.10391)| rnalm | single | alibi | 1026 |
|[BEACON-B512](https://arxiv.org/abs/2406.10391)| rnalm | single | alibi | 512 |
|[EcoRNA](https://github.com/cpa2001/Eco-RNA) | ecorna | single | rope (checkpoint-defined) | 1024 |



## 🔍 Usage
### Finetuning
To evalute on all RNA tasks, you can run the bash scripts in the `scripts` folder, for example:
```
cd RNABenchmark
bash ./scripts/BEACON-B/all_task.sh
```

### EcoRNA on NoncodingRNAFamily
This repository includes an opensource script for running EcoRNA and RNA-FM on the NoncodingRNAFamily task:

```bash
cd RNABenchmark
bash scripts/opensource/run_ncrna.sh ecorna
```

Release recommendation:
- Frozen / vanilla benchmark default: `weighted_layer_content`
- Strongest full fine-tune baseline: `cls_tanh`

Common runtime controls:
- `GPU_DEVICE` (e.g. `0,1,2,3`)
- `NPROC_PER_NODE` (e.g. `4`)
- `ECORNA_CHECKPOINT` (default: `../../output/ecorna-RNA-stage-d-100k`)
- `ECORNA_POOLING_STRATEGY` (`cls`, `cls_tanh`, `mean`, `cls_mean_concat`, `loop_mean_cls`, `content_mean`, `cls_ln`, `loop_mean_content`, `layer_weighted`, `loop_layer_attn_content`, `weighted_layer_content`, `weighted_cell_content`, `fixed_cell_content`, `fixed_cell_cls`)
- `ECORNA_NUM_LOOPS` (`1`, `2`, `3`, or `-1` to use checkpoint default)
- `SEED`

Example:
```bash
GPU_DEVICE=0,1,2,3 \
NPROC_PER_NODE=4 \
SEED=666 \
ECORNA_CHECKPOINT=../../output/ecorna-RNA-stage-d-100k \
ECORNA_POOLING_STRATEGY=weighted_layer_content \
ECORNA_NUM_LOOPS=3 \
bash scripts/opensource/run_ncrna.sh ecorna \
  --max_steps 1200 --save_steps 400 --eval_steps 200 --logging_steps 200
```

Strongest full fine-tune baseline:
```bash
GPU_DEVICE=0,1,2,3 \
NPROC_PER_NODE=4 \
SEED=666 \
ECORNA_CHECKPOINT=../../output/ecorna-RNA-stage-d-100k \
ECORNA_POOLING_STRATEGY=cls_tanh \
bash scripts/opensource/run_ncrna_ecorna_full_ft_best.sh
```

The test metrics are written to:
`outputs/ft/rna-all/NoncodingRNAFamily/ecorna/<pooling>/loops-<loops>/<seed>/results/ecorna_ncrna/test_results.json`

Plain readout validation helpers:
- `bash scripts/opensource/run_ncrna_plain_readout_pilot.sh`
- `python scripts/opensource/select_ncrna_plain_readout_winner.py --log_root <pilot_log_root> --output_json <winner.json>`
- `WINNER_JSON=<winner.json> bash scripts/opensource/run_ncrna_plain_readout_final.sh`
- `python scripts/opensource/summarize_ncrna_plain_readout.py --winner_json <winner.json> --output_json <summary.json>`

The plain readout launchers track `content_mean` and `cls_ln` under isolated variants, select the pilot winner using validation macro-F1 first (accuracy second), and print/validate EcoRNA tokenizer/config special token truth at startup.

Content-only cross-loop validation helpers:
- `bash scripts/opensource/run_ncrna_content_loop_pilot.sh`
- `python scripts/opensource/select_ncrna_content_loop_winner.py --log_root <pilot_log_root> --output_json <winner.json>`
- `WINNER_JSON=<winner.json> bash scripts/opensource/run_ncrna_content_loop_final.sh`
- `python scripts/opensource/summarize_ncrna_content_loop.py --winner_json <winner.json> --output_json <summary.json>`

Loop/layer diagnostic helper:
- `python scripts/opensource/inspect_ncrna_ecorna_features.py --include_loop_layer_content_grid ...`

Weighted-content validation helpers:
- Unified matrix runner:
  - `bash scripts/opensource/run_ncrna_weighted_content_matrix.sh phase0`
  - `PHASE0_JSON=<phase0.json> bash scripts/opensource/run_ncrna_weighted_content_matrix.sh pilot`
  - `WINNER_JSON=<winner.json> bash scripts/opensource/run_ncrna_weighted_content_matrix.sh final`
- Thin wrappers:
  - `bash scripts/opensource/run_ncrna_weighted_content_phase0.sh`
- `python scripts/opensource/summarize_ncrna_weighted_content_phase0.py --output_json <phase0.json>`
- `PHASE0_JSON=<phase0.json> bash scripts/opensource/run_ncrna_weighted_content_pilot.sh`
- `python scripts/opensource/select_ncrna_weighted_content_winner.py --output_json <winner.json>`
- `WINNER_JSON=<winner.json> bash scripts/opensource/run_ncrna_weighted_content_final.sh`
- `python scripts/opensource/summarize_ncrna_weighted_content_final.py --output_json <final_summary.json>`

Historical full-ft `cls_tanh` summary helper:
- `python scripts/opensource/summarize_ncrna_ecorna_full_ft_cls_tanh.py --output_json <full_ft_summary.json>`
### Computing embeddings
Embeddings from a dummy RNA sequence can be used as follows:

```python
import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from model.utrlm.modeling_utrlm import UtrLmModel
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

tokenizer = OpenRnaLMTokenizer.from_pretrained('./checkpoint/opensource/utr-lm-mrl', model_max_length=1026, padding_side="right", use_fast=True,)
model = UtrLmModel.from_pretrained('./checkpoint/opensource/utr-lm-mrl')     
sequences = ["AUUCCGAUUCCGAUUCCG"]
output = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="longest", max_length = 1026, truncation=True)
input_ids = output["input_ids"]
attention_mask = output["attention_mask"]

embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0] # shape [bz,length, hidden_size]
print(embedding.shape)
```



## License ##

This codebase is released under the Apache License 2.0 as in the [LICENSE](LICENSE) file.

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{ren2024beacon,
      title={BEACON: Benchmark for Comprehensive RNA Tasks and Language Models}, 
      author={Yuchen Ren and Zhiyuan Chen and Lifeng Qiao and Hongtai Jing and Yuchen Cai and Sheng Xu and Peng Ye and Xinzhu Ma and Siqi Sun and Hongliang Yan and Dong Yuan and Wanli Ouyang and Xihui Liu},
      year={2024},
      eprint={2406.10391},
      archivePrefix={arXiv},
      primaryClass={id='q-bio.QM' full_name='Quantitative Methods' is_active=True alt_name=None in_archive='q-bio' is_general=False description='All experimental, numerical, statistical and mathematical contributions of value to biology'}
}
```
