# GPTQ

This repository contains the code for the ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323). 
The current release includes the following features:

* An efficient implementation of the GPTQ algorithm: `gptq.py`
* Compressing all models from the OPT and BLOOM families to 2/3/4 bits, including weight grouping: `opt.py`, `bloom.py`, `zeroShot/`
* Evaluating the perplexity of quantized models on several language generation tasks: `opt.py`, `bloom.py`
* Evaluating the performance of quantized models on several ZeroShot tasks: `zeroShot/`
* A 3-bit quantized matrix full-precision vector product CUDA kernel: `quant_cuda_kernel.cu`, `quant_cuda.cpp`, `setup_cuda.py`
* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py`

## New Features

Update July 2023:

* Added `--static-groups` options which determines all group-grids in advance rather than dynamically during quantization, which has the effect that `--act-order` does not require any inference changes (that may cause slowdown) when used together with this option.

Together with the camera ready version of the paper we have added several updates to this repository:

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag `--new-eval`.
* Optimized 3bit kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via `--faster-kernel`.
* A minimal LlaMa integration (for more complete features see the [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) repository), which demonstrates two new tricks:`--act-order` (quantizing columns in order of decreasing activation size) and `--true-sequential` (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

Here is a summary of LLaMa results:

| Wiki2 PPL | FP16 | 4bit-RTN | 4bit-GPTQ | 3bit-RTN | 3bit-GPTQ | 3g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| LLaMa-7B  | 5.68 | 6.29     | **6.09**  | 25.54    | **8.07**  | 6.61       |
| LLaMa-13B | 5.09 | 5.53     | **5.36**  | 11.40    | **6.63**  | 5.62       |
| LLaMa-30B | 4.10 | 4.54     | **4.45**  | 14.89    | **5.69**  | 4.80       |
| LLaMa-65B | 3.53 | 3.92     | **3.84**  | 10.59    | **5.04**  | 4.17       |

Here is a sample command:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
```

The `--act-order` heuristic also dramatically improves accuracy on the OPT-66B outlier model: 9.55 to 9.34 and 14.16 to 9.95 PPL on Wiki2 for 4bit and 3bit, respectively.

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2 (the LLaMa integration currently requires a main install from source and `sentencepiece`)
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
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --wbits 3 --save opt175-3bit.pt
# Benchmark generating a 128 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --load opt175b-3bit.pt --benchmark 128
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
  journal={arXiv preprint arXiv:2210.17323}
}
```
