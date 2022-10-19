This folder contains code to reproduce our "Language Generation" experiments, namely:

* Compressing all models from the OPT and BLOOM model families to 3 or 4 bits.
* Evaluating perplexity of the quantized models.
* Our 3-bit kernel together with a small benchmarking script for individual matrix-vector products.

Code for the ZeroShot experiments and for running inference on compressed models will be released at a later date.

## Files

* `bloom.py`: script for running language generation experiments on BLOOM models
* `datautils.py`: utilities for handling datasets
* `gptq.py`: efficient implementation of the full GPTQ algorithm
* `modelutils.py`: some helper functions
* `opt.py`: script for running language generation experiments on OPT models
* `quant.py`: quantization utilities
* `quant_cuda.cpp`: PyTorch CUDA extension code
* `quant_cuda_kernel.cu`: CUDA kernel implemention
* `setup_cuda.py`: install 3-bit CUDA kernels
* `test_kernel.py`: test and benchmark 3-bit CUDA kernels

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html)

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.

# Language Generation

## OPT

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4
````

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.
For 175B you must request access from Meta and then convert it to a local HuggingFace checkpoint using their scripts in `metaseq`.
Once you have such a checkpoint, simply pass its path instead of `facebook/opt-125m`. 

## BLOOM

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4
````

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1.1b`, `bloom-1.7b`, `bloom-3b`, `bloom-7.1b`, `bloom`.

# 3-bit CUDA Kernels 

```
# Install kernels
python setup_cuda.py install
# Benchmark performance for FC2 layer of OPT-175B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py
```
