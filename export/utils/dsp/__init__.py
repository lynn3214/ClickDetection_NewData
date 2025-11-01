"""Dsp"""
"""
Digital Signal Processing utilities for dolphin click detection.
"""

from .filters import (
    BandpassFilter,
    HighpassFilter,
    PreEmphasisFilter,
    apply_bandpass,
    apply_highpass
)

from .envelope import (
    compute_hilbert_envelope,
    measure_envelope_width,
    compute_teager_kaiser,
    smooth_tkeo,
    compute_peak_factor,
    compute_energy_ratio
)

from .teager import (
    TeagerKaiserOperator
)

__all__ = [
    # Filters
    'BandpassFilter',
    'HighpassFilter',
    'PreEmphasisFilter',
    'apply_bandpass',
    'apply_highpass',
    
    # Envelope and TKEO
    'compute_hilbert_envelope',
    'measure_envelope_width',
    'compute_teager_kaiser',
    'smooth_tkeo',
    'compute_peak_factor',
    'compute_energy_ratio',
    'TeagerKaiserOperator',
]