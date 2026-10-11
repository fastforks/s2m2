# S2M2: faster inference fork

> **You are on `cuda-kernels`.** Research fork of [junhong-3dv/s2m2](https://github.com/junhong-3dv/s2m2) (base `8f3bd78`, 2026-04-28; `main` mirrors upstream). Two branches:
> - [`portable`](https://github.com/fastforks/s2m2/tree/portable): plain PyTorch, exports to ONNX / TensorRT, no build step.
> - [`cuda-kernels`](https://github.com/fastforks/s2m2/tree/cuda-kernels): `portable` plus custom CUDA kernels (prebuilt for sm_80/86/89/90/120; fp16-accumulate convs need `pip install nvidia-cudnn-frontend`).
>
> `portable` includes a correctness fix for upstream's +0.5 px lookup bug (S ETH3D EPE 0.315 -> 0.193), which costs a little speed in plain PyTorch.
> Licence, as upstream: code CC BY-NC 4.0 · weights CC BY-NC 4.0

S2M2-S, released weights, fp16 autocast, refine_iter 3; RTX 5060 Ti 16 GB, torch 2.14.1+cu130, TensorRT 11.4, 2026-10-10. Median of 10 runs; memory = torch peak allocated / reserved (TensorRT: context memory).

| version | KITTI ms | Midd-H ms | Midd-F ms | speed-up at H | peak MB at H (alloc / reserved) | peak MB at F (alloc / reserved) | startup / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|---|---|
| upstream | 89 | 340 | 1707 | 1.00x | 1700 / 2516 | 9003 / 12576 | 7.4 s / 2.7 s | 0.315 | 0.833 |
| **portable** | 91.8 | 350.3 | 1670 | 0.97x | 1742 / 2572 | 8195 / 12006 | 2.85 s / 28 ms | 0.193 | 0.652 |
| portable → TensorRT | 40.9 | 136.6 | 660 | 2.49x | ctx 949 | 8152 / 8778 | 1.6 s / none | 0.193 | 0.653 |
| **cuda-kernels** | 36.5 | 111.4 | 511 | 3.05x | 802 / 1322 | 2612 / 4428 | 2.2 s / 112 ms | 0.193 | 0.652 |

S2M2-XL:

| version | KITTI ms | Midd-H ms | Midd-F ms | speed-up at H | peak MB at H (alloc / reserved) | peak MB at F (alloc / reserved) | startup / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|---|---|
| upstream | 520 | 1765 | out of memory | 1.00x | 5241 / 7068 | out of memory | 11.3 s / 6.4 s | 0.190 | 0.553 |
| **portable** | 535.3 | 1817 | out of memory | 0.97x | 5324 / 7920 | out of memory | 3.0 s / 1 ms | 0.147 | 0.499 |
| portable → TensorRT | 237.2 | 767.5 | 3599 | 2.30x | ctx 2735 | ctx 10700 | 4.4 s / none | 0.150 | 0.502 |
| **cuda-kernels** | 213.6 | 683.8 | 3016 | 2.58x | 4000 / 4748 | 7328 / 9148 | 2.9 s / 31 ms | 0.149 | 0.499 |

---

<div align="center">
<h3>S<sup>2</sup>M<sup>2</sup>: Scalable Stereo Matching Model for Reliable Depth Estimation (ICCV 2025)</h3>
<h5>Junhong Min¹<sup>*</sup>, Youngpil Jeon¹, Jimin Kim¹, Minyong Choi¹</h5>
<p align="center">
  <a href="https://arxiv.org/abs/2507.13229">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2507.13229-b31b1b?style=for-the-badge&logo=arxiv">
  </a>
  <a href="https://junhong-3dv.github.io/s2m2-project">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-S²M²%20Page-brightgreen?style=for-the-badge&logo=googlechrome">
  </a>
</p>

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="fig/thumbnail.png" width="90%">)

[comment]: <> (</p>)

</div>


## 🤗 Notice
>This repository and its contents are **not related to any official Samsung Electronics products**.  
>All resources are provided **solely for non-commercial research and education purposes**.
---

## ✨ Key Features
### 🧩 Model
- Scalable stereo matching architecture  
- State-of-the-art performance on ETH3D (1st), Middlebury V3 (1st), and Booster (1st)
- Joint estimation of disparity, occlusion, and confidence
- Supports negative disparity estimation
- Optimal under the pinhole camera model with ideal stereo rectification (vertical disparity < 2px)

### ⚙️ Code
- ✅ FP16 / FP32 inference
- ✅ TorchScript/ONNX/TensorRT export
- ❌ Training pipeline (not included)

> Note: The publicly released model weights differ from the version used for Middlebury and ETH but are identical to the one used in Booster benchmark. This implementation replaces the dynamic attention-based refinement module with an UNet for stable ONNX export. It also includes an additional M variant and extended training data with transparent objects. 

---

## 🚀 Performance

> Detailed benchmark results and visualizations are available on the [Project Page](https://junhong-3dv.github.io/s2m2-project).

Inference Speed (FPS)** on **NVIDIA RTX 5090 (float16 + `refine_iter=3`):

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CH</th>
            <th>NTR</th>
            <th>Inference</th>
            <th>640x480</th>
            <th>1216x1024</th>
            <th>2432x2048</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>S</td>
            <td rowspan=2>128</td>
            <td rowspan=2>1</td>
            <td>torch.compile</td>
            <td>59.8</td>
            <td>26.4</td>
            <td>6.6</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>124.0</td>
            <td>59.4</td>
            <td>7.3</td>
        </tr>
        <tr>
            <td rowspan=2>M</td>
            <td rowspan=2>192</td>
            <td rowspan=2>2</td>
            <td>torch.compile</td>
            <td>32.4</td>
            <td>12.5</td>
            <td>3.1</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>66.7</td>
            <td>18.3</td>
            <td>3.8</td>
        </tr>
        <tr>
            <td rowspan=2>L</td>
            <td rowspan=2>256</td>
            <td rowspan=2>3</td>
            <td>torch.compile</td>
            <td>25.8</td>
            <td>7.6</td>
            <td>2.0</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>46.6</td>
            <td>11.2</td>
            <td>2.4</td>
        </tr>
        <tr>
            <td rowspan=2>XL</td>
            <td rowspan=2>384</td>
            <td rowspan=2>3</td>
            <td>torch.compile</td>
            <td>16.3</td>
            <td>4.3</td>
            <td>1.1</td>
        </tr>
        <tr>
            <td>TensorRT</td>
            <td>26.6</td>
            <td>6.4</td>
            <td>1.4</td>
        </tr>
    </tbody>
</table>

---

## 🔧 Installation

We recommend using Python 3.10, PyTorch 2.9, CUDA 12.9, CUDNN 9.1.0, and tensorRT 10.13.3 with Anaconda.

```bash
git clone https://github.com/junhong-3dv/s2m2
cd s2m2
```
```bash
conda env create -n s2m2 -f environment.yml
conda activate s2m2
```
```bash
pip install -e .
```


If the environment setup via .yml doesn’t work smoothly,
you can manually install the main dependencies with:
```bash
pip install torch torchvision opencv-python open3d onnx onnxruntime-gpu onnxscript tensorrt-cu12==10.13.3.9 --extra-index-url https://pypi.nvidia.com/tensorrt-cu12-libs

```
That should cover most of the required packages for running the demo.

## 🚀 Pre-trained Models and Inference

### 1. Download Pre-trained Models

Create a directory for weights and download the desired models from the links below.

```bash
mkdir weights
mkdir weights/pretrain_weights
```

| Model | Download | Model Size |
| :---: | :---: | :--: |
| **S** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth) | 26.5M | 
| **M** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH192NTR2.pth) | 80.4M | 
| **L** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH256NTR3.pth) | 181M | 
| **XL**| [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth) | 406M | 

### 2. Run Basic Demo
To generate a result for a single input, run `demo/visualize_2d_simple.py`.

```bash
python ./demo/visualize_2d_simple.py --model_type XL --num_refine 3 
```
| arg | default | type | help |
| :---: | :---: | :--: | :--: |
| --model_type | 'XL' | str | select model type: [S,M,L,XL] |
| --num_refine | 3 | int | number of local iterative refinement |
| --torch_compile | False | set_true | apply torch_compile  | 
| --allow_negative | False | set_true | allow negative disparity  | 


### 3. Run 3D Visualization Demo

To visualize the 3D output interactively, run `demo/visualize_3d_booster.py` or `demo/visualize_3d_middlebury.py`

```bash
python ./demo/visualize_3d_booster.py --model_type L 
```
For 'visualize_3d_middlebury.py --model_type XL ', result should be like below. 
<p align="center">
  <img src="fig/result_bicycle2.png" width="90%">
</p>
If you failed to reproduce this, let me know. 

## 🚀 Model Optimization (ONNX / TensorRT)

### 1. Export to ONNX (OPSET=18)

Use export_onnx.py to convert the model to ONNX:

```console
python demo/export_onnx.py --model_type $MODEL_TYPE --img_width $IMG_WIDHT --img_height $IMG_HEIGHT
```
### 2. Export to TensorRT (tested with 10.13.3)
Use export_tensorrt.py to build a TensorRT engine:

```console
python demo/export_tensorrt.py --model_type $MODEL_TYPE --img_width $IMG_WIDHT --img_height $IMG_HEIGHT --precision $PRECISION
```
or simply in the terminal
```console
trtexec --onnx={onnx_file_path} --saveEngine=./{trt_file_path} --fp16 --precisionConstraints=obey --layerPrecisions=node_linalg_vector_norm_2:fp32
```
Supported TensorRT precisions: fp32, tf32, fp16

## 📜 Citation

If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{min2025s2m2,
  title={{S\textsuperscript{2}M\textsuperscript{2}}: Scalable Stereo Matching Model for Reliable Depth Estimation},
  author={Junhong Min and Youngpil Jeon and Jimin Kim and Minyong Choi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
