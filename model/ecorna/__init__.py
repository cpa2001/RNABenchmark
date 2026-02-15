"""EcoRNA model integration for BEACON benchmark."""

from .modeling_ecorna import (
    EcoRNAForSequenceClassification,
    EcoRNAForNucleotideLevel,
)

__all__ = [
    "EcoRNAForSequenceClassification",
    "EcoRNAForNucleotideLevel",
]
