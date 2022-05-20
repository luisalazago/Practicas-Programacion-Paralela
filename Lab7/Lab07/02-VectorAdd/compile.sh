#!/bin/sh
#
# This script will invoke the nvcc compiler driver to illustrate code
# to be executed on a CUDA-capable GPU (.ptx) as well as code meant
# to be executed on the host CPU (.cu.cpp.ii) as well as code meant
#
clear
echo "Generating PTX code from HelloWorld.cu: (HelloWorld.ptx)"
nvcc -ptx HelloWorld.cu
echo "Generating host code from HelloWorld.cu: (HelloWorld.cu.cpp.ii)"
nvcc -cuda HelloWorld.cu
echo "Generating exceutable image from HelloWorld.cu: (HelloWorld)"
nvcc HelloWorld.cu -o HelloWorld
echo -e "Done!\n"
