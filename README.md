# GPTQ

This repository contains the code for the paper *GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers*. 
The current release includes the following features:

* An efficient implementation of the GPTQ algorithm: `gptq.py`
* Compressing all models from the OPT and BLOOM families to 2/3/4 bits, including weight grouping: `opt.py`, `bloom.py`, `zeroShot/`
* Evaluating the perplexity of quantized models on several language generation tasks: `opt.py`, `bloom.py`
* Evaluating the performance of quantized models on several ZeroShot tasks: `zeroShot/`
* A 3-bit quantized matrix full-precision vector product CUDA kernel: `quant_cuda_kernel.cu`, `quant_cuda.cpp`, `setup_cuda.py`
* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py`

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.4)

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.

## Language Generation

### OPT

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 [--groupsize 1024]
````

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.
For the 175B-parameter mode, you have to request access from Meta and then convert it to a local HuggingFace checkpoint using their scripts in `metaseq`.
Once you have such a checkpoint, simply pass its path instead of `facebook/opt-125m`. 

### BLOOM

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 [--groupsize 1024]
````

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`, `bloom`.

## ZeroShot

See `zeroShot/` folder.

## 3-bit CUDA Kernels 

```
# Install kernels
python setup_cuda.py install

# Benchmark performance for FC2 layer of OPT-175B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py

# Benchmark language generation with 3-bit OPT-175B:
# OPT175B denotes the name of the folder with the HuggingFace OPT-175b checkpoint (see above)

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --wbits 3 --save opt66-3bit.pt
# Benchmark generating a 128 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --load opt66-3bit.pt --benchmark 128
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python opt.py OPT175B c4 --benchmark 128
```

Please note that our 3-bit kernels are currently only optimized for OPT-175B running on 1xA100 or 2xA6000 and may thus yield suboptimal performance on smaller models or on other GPUs.

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:XXXX.XXXX}
}
```
