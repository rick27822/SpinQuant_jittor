# coding=utf-8
"""Single-GPU Jittor SpinQuant implementation.

Includes rotation learning, RTN/GPTQ weight fake quantization and WikiText-2 PPL.
Distributed training, integer inference kernels and cache generation
are outside the supported scope. Subpackages intentionally have empty init files.
"""
