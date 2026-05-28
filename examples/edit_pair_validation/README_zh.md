# 所有 i2i / image-edit 模型横向验证

目标：验证仓库里能做图像编辑或图生图的模型，是否能从第一张 Amazon 商品图生成第二张广告图风格的结果。

关键约束：

- 生成阶段只允许使用 `source.jpg + prompt + seed`。
- `target.jpg` 只在生成后用于评分和并排可视化。
- prompt 可以写目标图新增的大标题和卖点文字，因为这些是“要编辑出来的新内容”。
- prompt 不抄 source 管身上的原始包装小字，只要求保持它们，用来检查 reference/IP/label preservation。

这避免了“把答案给模型再验证模型会不会复读”的问题。

## 1. 下载图片

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py prepare \
  --output-dir data/edit_pair_validation/amazon_lipcare
```

会生成：

```text
data/edit_pair_validation/amazon_lipcare/source.jpg
data/edit_pair_validation/amazon_lipcare/target.jpg
```

两张图都是 `500x500`，这个任务比高分辨率商品图更苛刻。

## 2. 查看支持的模型

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py list-models
```

当前脚本纳入了仓库中明确能吃参考图并输出单张图的模型：

```text
qwen_image_edit
qwen_image_edit_2509
qwen_image_edit_2511
qwen_image_edit_2511_lightning
firered_image_edit_1_0
firered_image_edit_1_1
joyai_image_edit
hidream_o1_image
hidream_o1_image_dev
z_image_omni_base
flux1_kontext_dev
step1x_edit
nexus_gen_editing
flux2_dev
flux2_klein_base_4b
flux2_klein_4b
flux2_klein_base_9b
flux2_klein_9b
flux2_template_edit_4b
anima_img2img
```

没有混入视频 I2V、音频、纯 depth/canny/controlnet、inpaint-only、upscale-only、style-LoRA 模型，因为这些任务定义不同，不适合放在同一主榜里。仓库当前的 SD 1.5 / SDXL pipeline 公开 `__call__` 没有暴露 `input_image` 参数，所以也不放进 i2i 榜。

## 3. 先单独跑一个模型

建议先用一个 seed 验证环境：

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py generate \
  --model qwen_image_edit_2511 \
  --source data/edit_pair_validation/amazon_lipcare/source.jpg \
  --output-dir outputs/edit_pair_validation/all_i2i \
  --seed 0 \
  --height 1024 --width 1024 \
  --device cuda --dtype bfloat16
```

再评估：

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py compare \
  --source data/edit_pair_validation/amazon_lipcare/source.jpg \
  --target data/edit_pair_validation/amazon_lipcare/target.jpg \
  --generated outputs/edit_pair_validation/all_i2i/qwen_image_edit_2511/seed0.png \
  --output-dir outputs/edit_pair_validation/all_i2i/qwen_image_edit_2511/eval_seed0
```

评估目录会有：

```text
comparison_full.png
comparison_new_ad_copy_left.png
comparison_product_right.png
comparison_tube_label_right.png
metrics.json
```

数值指标只衡量和目标图布局/像素的接近程度；管身 IP、小字是否保持，必须看 `comparison_product_right.png` 和 `comparison_tube_label_right.png`，最好再接 OCR。

## 4. 跑全部模型

为了避免显存残留，`run-all` 会每个模型单独开子进程跑：

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py run-all \
  --source data/edit_pair_validation/amazon_lipcare/source.jpg \
  --target data/edit_pair_validation/amazon_lipcare/target.jpg \
  --output-dir outputs/edit_pair_validation/all_i2i \
  --models all_relevant \
  --seeds 0 \
  --height 1024 --width 1024 \
  --device cuda --dtype bfloat16
```

更 solid 的版本至少跑固定 8 个 seed：

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py run-all \
  --source data/edit_pair_validation/amazon_lipcare/source.jpg \
  --target data/edit_pair_validation/amazon_lipcare/target.jpg \
  --output-dir outputs/edit_pair_validation/all_i2i_seed0_7 \
  --models all_relevant \
  --seeds 0 1 2 3 4 5 6 7 \
  --height 1024 --width 1024 \
  --device cuda --dtype bfloat16
```

如果你想先跑小集合：

```bash
python examples/edit_pair_validation/all_i2i_reference_to_target.py run-all \
  --source data/edit_pair_validation/amazon_lipcare/source.jpg \
  --target data/edit_pair_validation/amazon_lipcare/target.jpg \
  --output-dir outputs/edit_pair_validation/top_candidates \
  --models qwen_image_edit_2511 flux1_kontext_dev flux2_klein_base_4b z_image_omni_base joyai_image_edit \
  --seeds 0 1 2 \
  --height 1024 --width 1024 \
  --device cuda --dtype bfloat16
```

输出汇总：

```text
outputs/edit_pair_validation/.../results.jsonl
outputs/edit_pair_validation/.../summary.csv
```

## 5. 结论应该怎么写

不要只写“哪个最好看”。建议每个模型、每个 seed 人工或 OCR 记录四项：

| 项目 | 判据 |
| --- | --- |
| layout_success | 是否接近目标广告构图：左文案、右产品、夹具、滴液 |
| new_copy_readable | 新增标题和卖点是否可读、拼写正确 |
| source_label_preserved | 原管身品牌/包装小字是否保持，没有换品牌或乱写 |
| product_identity_preserved | 黄色管体、蜂蜜护唇产品的视觉身份是否保持 |

最后统计：

```text
layout_success / N
new_copy_readable / N
source_label_preserved / N
product_identity_preserved / N
all_pass / N
```

像素 PSNR/MAE 只能辅助排序“像不像目标布局”，不能证明小字保持。小字/IP 保持需要 OCR 或人工可读性标注。

## 6. 常见问题

`nexus_gen_editing` 要求 `transformers==4.49.0`，如果环境版本不对，该模型会失败，但其他模型会继续跑。

`flux2_dev` 默认带 CPU/GPU offload 配置，速度会慢一些，但更稳。

`anima_img2img` 是传统 img2img baseline，不是现代 instruction edit 模型。它放进来是为了给底线参考，不应和 Qwen/Flux Kontext 这类模型做同等能力假设。
