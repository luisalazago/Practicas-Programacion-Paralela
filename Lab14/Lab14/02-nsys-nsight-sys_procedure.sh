#!/usr/bin/env bash
#
# 02-nsys-nsight-sys_procedure.sh
#
# Demonstrates using
#   nsys (NVIDIA Nsight Systems), a command-line system-level profiler
#
# Available on GPU Pascal architectures onwards (2016-)
#

nsys status -e

nsys profile --stats=true ./CUDA_vectorAddition_blocks

nsys profile --stats=true --force-overwrite=true -o CUDA_vectorAddition_blocks-%h-nsys.qdrep ./CUDA_vectorAddition_blocks

nsight-sys &
