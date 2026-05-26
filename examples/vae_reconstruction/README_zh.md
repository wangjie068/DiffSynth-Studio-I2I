# Phase 0: VAE Reconstruction 对比实验

这个实验只执行：

```text
原图 x -> VAE encode -> VAE decode -> x_recon
```

没有 prompt、DiT、采样器或去噪步骤。因此，如果重建图中的小字已经模糊、变形或消失，可以把损失归因到对应 VAE 的编码/解码瓶颈，而不是扩散生成过程。

## 对比模型

脚本默认依次测试以下模型中实际使用的 VAE/AE 权重，且每次只把一个 VAE 加载到显存：

| 参数 | 权重来源 | 说明 |
| --- | --- | --- |
| `qwen_image` | `Qwen/Qwen-Image` | `Qwen-Image` 及 `Qwen-Image-Edit` 使用的 VAE |
| `flux1` | `black-forest-labs/FLUX.1-dev` | FLUX.1 AE |
| `flux2` | `black-forest-labs/FLUX.2-klein-4B` | FLUX.2 VAE |
| `z_image` | `Tongyi-MAI/Z-Image-Turbo` | Z-Image VAE |
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL VAE |
| `sd15` | `AI-ModelScope/stable-diffusion-v1-5` | SD 1.5 VAE |

SDXL 与 SD 1.5 的 encoder 会输出一个分布。此实验固定使用 posterior mean，而不是随机 sample，避免随机噪声影响 VAE 间的可重复对照。

## 服务器安装

在仓库根目录执行：

```bash
pip install -e .
```

脚本默认通过 ModelScope 下载，而且只下载上表的 VAE 权重文件，不会下载 text encoder 或 diffusion transformer。

如需先单独下载全部 VAE：

```bash
python examples/vae_reconstruction/compare_vae_reconstruction.py \
  --download-only \
  --model-dir ./models
```

脚本也保留了 `--download-source huggingface` 接口，但上表模型 ID 依据本仓库的 ModelScope 配置填写。只有在确认相同 ID 和文件结构存在于 Hugging Face 后，才应切换下载源：

```bash
--download-source huggingface
```

## 运行实验

以下命令已使用指定图片 URL 作为默认输入：

```bash
python examples/vae_reconstruction/compare_vae_reconstruction.py \
  --output-dir outputs/phase0_vae_all \
  --model-dir ./models \
  --device cuda \
  --dtype float32 \
  --strict
```

`float32` 更适合作为对照实验的默认精度。如果显存不够，可以改用 `--dtype bfloat16`，但报告中应注明计算精度变化。

只测试当前 Qwen-Image-Edit 路径所用的 VAE：

```bash
python examples/vae_reconstruction/compare_vae_reconstruction.py \
  --vae qwen_image \
  --output-dir outputs/phase0_qwen_vae \
  --device cuda \
  --dtype float32
```

## 小字区域量化

首次运行先打开 `outputs/phase0_vae_all/input_original.png`，读取需要论证的小字区域坐标 `(x0, y0, x1, y1)`。再次运行时通过 `--roi` 填入坐标，可重复添加多个文字区域：

```bash
python examples/vae_reconstruction/compare_vae_reconstruction.py \
  --output-dir outputs/phase0_vae_text_rois \
  --device cuda \
  --dtype float32 \
  --roi text_top:X0,Y0,X1,Y1 \
  --roi text_bottom:X0,Y0,X1,Y1 \
  --strict
```

将示例中的大写坐标替换为实际整数，例如 `--roi text_top:120,80,480,145`。

## 输出文件

| 文件 | 用途 |
| --- | --- |
| `input_original.png` | 实验原图 |
| `recon_MODEL.png` | 各 VAE 重建结果 |
| `comparison_full.png` | 原图与所有重建图并排对比 |
| `error_x4_MODEL.png` | 像素绝对误差可视化，亮处为改动更大区域 |
| `comparison_roi_NAME.png` | 指定小字 ROI 的放大并排对比 |
| `metrics.csv` | 全图和 ROI 的 MAE、RMSE、PSNR |
| `patch_metrics.csv` | 默认 64x64 patch 的误差排行，用于辅助定位退化集中区 |
| `manifest.json` | 输入 padding、权重来源、latent shape 与失败信息 |

## 实验约束

- 所有 VAE 使用同一张原图、同一精度和同一评价方式。
- 脚本不缩放原图。若图像尺寸不满足 VAE 倍数要求，仅在右侧和底部复制边缘像素 padding，重建后再裁回原始尺寸。
- 不将 `error_x4_*.png` 当作定量指标；其亮度已放大方便观察。定量结论应引用 `metrics.csv` 中的小字 ROI 指标及对应重建裁图。
- 正式对比命令使用 `--strict`，以避免权重下载或加载失败时留下不完整却被误读的横向结果。
