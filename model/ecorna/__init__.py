"""EcoRNA model integration for BEACON benchmark."""

from .modeling_ecorna import (
    EcoRNAForSequenceClassification,
    EcoRNAForNucleotideLevel,
    EcoRNALayerWeightedPooler,
)

__all__ = [
    "EcoRNAForSequenceClassification",
    "EcoRNAForNucleotideLevel",
    "EcoRNALayerWeightedPooler",
]
