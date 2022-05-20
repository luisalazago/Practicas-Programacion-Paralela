#!/usr/bin/env bash
#
# 03-nv-nsight_procedure.sh
#
# Demonstrates using
#   nv-nsight-cu* (NVIDIA Nsight Compute, CLI and GUI versions)
#
# Available on GPU Turing and Volta architectures onwards onwards (Aug 2017-)
#

sudo /usr/local/cuda-11.1/bin/nv-nsight-cu-cli --section ".*" -o CUDA_vectorAddition_blocks-%h -f CUDA_vectorAddition_blocks

nv-nsight-cu &
