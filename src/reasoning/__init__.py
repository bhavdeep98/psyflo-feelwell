"""
Reasoning Module - Deep Clinical Analysis

This module implements the "Hidden Clinician" logic using Mistral-7B
for non-linear pattern detection and clinical metric assessment.

Components:
- MistralReasoner: Structured reasoning with Mistral-7B
- ClinicalMetrics: Seven-dimension clinical assessment
- ReasoningResult: Immutable reasoning output with trace

SLA: <2s reasoning latency (with timeout fallback)
"""

from .mistral_reasoner import MistralReasoner, ReasoningResult
from .clinical_metrics import ClinicalMetrics, MetricScore

__all__ = [
    'MistralReasoner',
    'ReasoningResult',
    'ClinicalMetrics',
    'MetricScore',
]
