# DDA5001 Final Project Part I 中文指南

本文档用于在学校电脑上快速理解、运行和保存项目结果。正式截止时间和提交规则以 Blackboard 和课程 handbook 为准。

## 1. Project 要做什么

Part I 要在 Shakespeare 字符数据集上训练 GPT-2 Medium，并比较三种优化器：

- AdamW
- Muon
- 带动量的 SGD（SGDM）

每种优化器都要使用一个默认学习率和一个自行选择的学习率，因此一共需要六组正式实验。

| Optimizer | 默认学习率 | 本项目选择的第二个学习率 |
| --- | ---: | ---: |
| AdamW | 0.0003 | 0.0006（2 倍） |
| Muon | 0.01 | 0.02（2 倍） |
| SGDM | 0.1 | 0.2（2 倍） |

正式配置为 5,000 次更新，总计 92,160,000 个训练 prediction tokens，使用 cosine learning-rate schedule。

## 2. 代码部分

课程要求只补全 `p1/model.py`、`p1/train.py` 和 `p1/plot.py` 中的七个 TODO：

1. GPT causal attention mask；
2. next-token label 向后移动一个 sequence position；
3. 在完成梯度累积和裁剪后执行 `optimizer.step()`；
4. 配置原生 `torch.optim.AdamW`；
5. 配置原生 `torch.optim.SGD`；
6. 配置原生 `torch.optim.Muon` 和辅助 AdamW；
7. 使用给定的 x/y 数据绘制 loss curve。

本仓库已完成这七处修改。代码已通过官方 `check_solution.py`、CPU smoke test，以及三种 optimizer 的小型批量测试。

## 3. 在 VS Code 中下载项目

在 VS Code 选择 **Terminal → New Terminal**，然后运行：

```bash
cd ~
git clone https://github.com/iruby1/ml.git
cd ml/p1
```

如果已经 clone 过：

```bash
cd ~/ml
git pull
cd p1
```

确认代码版本：

```bash
git log -2 --oneline
git status
```

## 4. 启用学校 Python 环境

```bash
source /opt/dda5001/venv/bin/activate
```

检查 Python、PyTorch、Muon 和 GPU：

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Muon:', hasattr(torch.optim, 'Muon'))"
nvidia-smi
```

环境应满足：

- PyTorch 2.9 或更高版本；
- `CUDA: True`；
- `Muon: True`；
- `nvidia-smi` 能显示 NVIDIA GPU。

如果 `/opt/dda5001/venv/bin/activate` 不存在，先向老师或机房管理员确认学校环境路径，不要直接改动全局 Python。

## 5. 正式训练前的检查

必须先运行：

```bash
python check_solution.py
python train.py config/smoke.py --device=cpu --out_dir=checks/school-cpu
python setup/verify_env.py
```

`check_solution.py` 没有报错即为通过。Smoke test 最后应显示 `Completed`。

## 6. 使用 tmux 防止断线后训练停止

先检查：

```bash
tmux -V
```

如果 tmux 存在：

```bash
tmux new -s dda5001
cd ~/ml/p1
source /opt/dda5001/venv/bin/activate
```

训练开始后，按 `Ctrl+B`，松开后再按 `D` 可以离开但保持训练。重新进入：

```bash
tmux attach -t dda5001
```

## 7. GPU 显存足够时运行六组实验

默认配置约需 20 GiB GPU 显存。一张 GPU 使用 `--jobs 1`：

```bash
python run_experiments.py \
  --config config/train_shakespeare_char.py \
  --optimizers adamw,muon,sgdm \
  --multipliers 1,2 \
  --protocol tokens \
  --devices 0 \
  --jobs 1 \
  --output runs/final-sweep
```

该命令会顺序完成：

```text
00-adamw-tokens-lr1
01-adamw-tokens-lr2
02-muon-tokens-lr1
03-muon-tokens-lr2
04-sgdm-tokens-lr1
05-sgdm-tokens-lr2
```

只有一张 GPU 时不要将 `--jobs` 改成 2。

## 8. GPU 显存不足时

复制一份配置：

```bash
cp config/train_shakespeare_char.py config/school.py
```

用 VS Code 修改 `config/school.py`：

```python
batch_size = 8
gradient_accumulation_steps = 9
eval_iters = 72
```

如果 `torch.compile` 报错，再设置：

```python
compile = False
```

先进行一次小型 GPU 测试：

```bash
python train.py config/school.py --max_tokens=18432 --out_dir=checks/school-gpu-test
```

成功后用同一配置运行六组实验：

```bash
python run_experiments.py \
  --config config/school.py \
  --optimizers adamw,muon,sgdm \
  --multipliers 1,2 \
  --protocol tokens \
  --devices 0 \
  --jobs 1 \
  --output runs/final-sweep
```

六组正式实验必须使用相同的 batch size、gradient accumulation、evaluation size 和 compile 设置，并在报告中说明硬件配置修改。

## 9. 检查训练是否全部完成

```bash
find runs/final-sweep -name summary.json -print
```

应该存在六个 `summary.json`。查看每组状态和最终 validation loss：

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path('runs/final-sweep').glob('*/summary.json')):
    data = json.loads(path.read_text())
    print(
        path.parent.name,
        'status=', data.get('status'),
        'val_loss=', data.get('endpoint_scores', {}).get('loss'),
        'tokens=', data.get('tokens'),
    )
PY
```

每组都应为 `status=completed`，并且 `tokens=92160000`。

## 10. 生成图表

为六组实验分别生成图表：

```bash
for d in runs/final-sweep/[0-9][0-9]-*; do
  python plot.py --run-dir="$d"
done
```

每组会得到：

```text
loss_vs_tokens.png
loss_vs_train_seconds.png
```

生成六组实验的综合比较图：

```bash
python plot.py --runs-dir=runs/final-sweep
```

综合图位于：

```text
runs/final-sweep/comparison_tokens_vs_tokens.png
runs/final-sweep/comparison_tokens_vs_train_seconds.png
```

## 11. 生成 Shakespeare 文本样本

从 validation loss 较好的一组中选择 `best.pt`。例如：

```bash
python sample.py \
  --checkpoint=runs/final-sweep/02-muon-tokens-lr1/best.pt \
  --device=cuda \
  --tokens=400 \
  --output=runs/final-sweep/02-muon-tokens-lr1/samples.txt
```

如果其他实验的 validation loss 更低，将 checkpoint 路径替换为对应目录。

## 12. 报告需要讨论什么

报告应根据实际曲线和生成文本回答：

- 哪种优化器收敛最快；
- 在相同 token 数量下，哪种方法的 validation loss 最低；
- 在相同训练时间下，哪种方法表现最好；
- 学习率从默认值变成 2 倍后，每种优化器如何变化；
- 曲线是否出现 crossover；
- 是否出现震荡、不稳定或过拟合；
- 生成文本是否具有 Shakespeare 风格，是否连贯；
- 如果修改了默认硬件配置，说明修改原因和具体参数。

不要只说“某个 optimizer 最好”；需要说明是在什么学习率、什么训练阶段、什么比较标准下更好。

## 13. 导出提交结果

```bash
python export_results.py \
  --runs-dir=runs/final-sweep \
  --output=results01.zip
```

检查压缩包：

```bash
ls -lh results01.zip
unzip -t results01.zip
```

`results01.zip` 会包含代码、配置、metrics、日志、文本样本和图片，但不会包含大型 checkpoint。报告 PDF 完成后也需要放入最终提交包。

## 14. 离开学校电脑前必须保存的内容

将完整实验目录打包：

```bash
tar -czf final-sweep-with-checkpoints.tar.gz runs/final-sweep
```

必须下载或复制：

```text
results01.zip
final-sweep-with-checkpoints.tar.gz
```

其中第二个文件包含 `best.pt` 和 `last.pt`。Checkpoint 不要推送到 GitHub，因为文件很大，而且项目 `.gitignore` 已经排除它们。

## 15. 最简操作清单

```text
1. clone 或 git pull
2. source /opt/dda5001/venv/bin/activate
3. python check_solution.py
4. CPU smoke test
5. python setup/verify_env.py
6. nvidia-smi 检查显存
7. 在 tmux 中执行六组训练
8. 确认六个 summary.json 都是 completed
9. 生成独立图和综合图
10. 生成 400-token 文本
11. 导出 results01.zip
12. 下载 results01.zip 和完整 checkpoint 压缩包
```

## 16. 项目原始说明

- [Part I 官方 README](p1/README.md)
- [Part I coding-agent contract](p1/AGENTS.md)
- [个人 GPU 和小显存说明](p1/setup/README.md)

