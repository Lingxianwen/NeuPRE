"""
NeuPRE: Neuro-Symbolic Protocol Reverse Engineering

A state-of-the-art protocol reverse engineering framework combining:
- Information Bottleneck for format learning
- Deep Kernel Learning for state exploration
- Neuro-Symbolic reasoning for constraint inference
"""

__version__ = '1.0.0'
__author__ = 'NeuPRE Team'

from .neupre import NeuPRE, setup_logging
from .modules import (
    InformationBottleneckFormatLearner,
    DeepKernelStateExplorer,
    NeuroSymbolicLogicRefiner
)
from .utils import NeuPREEvaluator

__all__ = [
    'NeuPRE',
    'setup_logging',
    'InformationBottleneckFormatLearner',
    'DeepKernelStateExplorer',
    'NeuroSymbolicLogicRefiner',
    'NeuPREEvaluator',
]
