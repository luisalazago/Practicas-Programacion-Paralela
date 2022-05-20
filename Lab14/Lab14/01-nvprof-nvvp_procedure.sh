#!/usr/bin/env bash
#
# 01-nvprof-nvvp_procedure.sh
#
# Demonstrates using
#   nvprof, a light-weight command-line CUDA profiler
#   nvvp, the NVIDIA Visual Profiler
#

nvprof ./CUDA_vectorAddition_blocks

nvprof ./CUDA_vectorAddition_threads

nvprof ./CUDA_vectorAddition_blocks_threads

nvprof --print-gpu-trace ./CUDA_vectorAddition_blocks

nvprof --print-gpu-trace ./CUDA_vectorAddition_threads

nvprof --print-gpu-trace ./CUDA_vectorAddition_blocks_threads

nvprof --log-file CUDA_vectorAddition_blocks-%h-smry.nvprof ./CUDA_vectorAddition_blocks

nvprof --print-gpu-trace --log-file CUDA_vectorAddition_blocks-%h-trc.nvprof ./CUDA_vectorAddition_blocks

nvprof --analysis-metrics --fo CUDA_vectorAddition_blocks-%h-mtrix.nvprof ./CUDA_vectorAddition_blocks

nvvp &
