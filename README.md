# ARCJ / TMCHT 复现

复现论文 **《A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns》**
（arXiv:2410.16155v2）。包含两部分：

1. **TMCHT**：大规模、多智能体、多拓扑、**独立记忆**的文本攻击评测框架。一个攻击者 agent
   通过一对一对话，试图误导整个 agent 社会。
2. **ARCJ**：两阶段"对抗自复制传染越狱"方法。
   - **Stage 1（检索后缀）**：让毒样本更容易被 DPR 检索到（论文 Eq.8）。
   - **Stage 2（复制后缀）**：迫使模型自我复制毒样本，缓解"toxicity disappearing"，使其具备
     传染能力（论文 Eq.9）。两种模式：`global`（通用后缀）/ `single`（每样本后缀）。

> 说明：本仓库默认使用**开源免授权**模型（`Qwen/Qwen2.5-7B-Instruct`）作为论文 Llama3-8B 的替代，
> 数据为符合论文格式的内置样例（+ 生成脚本）。因此复现的是**方法与趋势**，绝对数值会与论文有差异。

---

## 1. 环境安装（Linux + Anaconda）

```bash
conda env create -f environment.yml
conda activate arcj
pip install -e .          # 可选：安装为可编辑包（否则脚本用内置 sys.path 引导）
```

- 需要 **CUDA GPU（建议 24GB+ 显存）** 来跑 7B 模型 + GCG 优化。
- 显存不足时：在 config 里把 `model.load_in_4bit: true`（需 `bitsandbytes`），或换更小的模型。
- `environment.yml` 默认装 CUDA 版 PyTorch；纯 CPU 只能跑流程验证，无法实跑大模型/GCG。
- **torch 版本**：`transformers>=5` 出于安全限制（CVE-2025-32434）不再用 `torch.load` 加载
  `.bin` 权重，需 `torch>=2.6`，但 `safetensors` 权重不受此限。代码默认 `model.use_safetensors: true`
  强制走 safetensors（本仓库用到的 Qwen / DPR 模型都有 safetensors），所以 `torch<2.6` 也能跑。
  若你切换到只有 `.bin` 的模型，请升级到 `torch>=2.6`。

### 显存 OOM 怎么办（CUDA out of memory）

7B 模型（约 15GB）+ GCG 的 Stage 2 复制后缀优化最吃显存。代码已用 `logits_to_keep`
只计算目标尾部的 logits（而非整段 `[B, L, 151936]`）来省显存。若仍 OOM，按需调：

- 调小 `gcg.eval_chunk`（候选评估的分块大小，如 `8`）——峰值显存主要由它决定；必要时也调小 `gcg.batch_size`。
- 设环境变量减少碎片：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 选一张空闲卡：`CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py ...`（多卡机器上别人占了 GPU0 时尤其有用）。
- 显存实在不够：config 里 `model.load_in_4bit: true`（需 `bitsandbytes`），模型降到约 5GB。

## 2. 数据

内置 `data/topics.json`、`data/questions.json`（论文格式）。扩充到 100 条见 [`data/README.md`](data/README.md)：

```bash
export OPENAI_API_KEY=sk-...
python scripts/generate_data.py --num-topics 100 --model gpt-4o --seed-existing
```

## 3. 快速验证

不需要 GPU，先跑单元测试（拓扑/指标/记忆/数据/GCG 优化器循环/stub 模拟）：

```bash
pytest                    # 全部测试
```

有 GPU 后，跑端到端 smoke（小模型，几分钟）：

```bash
python scripts/run_experiment.py --config configs/smoke.yaml
```

它会完整走一遍：建拓扑 → ARCJ 优化检索/复制后缀 → 多轮一对一对话 → 周期评测 ASR → 存结果。

## 4. 复现论文实验

```bash
# 单个实验（按 config + 命令行覆盖；--reuse-suffix 可跨密度/拓扑复用同一套后缀）
python scripts/run_experiment.py --config configs/structure_line.yaml \
    --attack arcj --density 0.5 --reuse-suffix --name structure_line_arcj_d0.5

# 一键跑全部（结构表 1 + 规模表 2 + 毒性消失图）。很耗算力！
MODEL=Qwen/Qwen2.5-7B-Instruct DEVICE=cuda bash scripts/run_all.sh
QUICK=1 bash scripts/run_all.sh          # 快速小规模过一遍

# 毒性消失实验（图 4 / 7）
python scripts/run_toxicity.py --config configs/toxicity_disappearing.yaml
```

出图 / 出表：

```bash
python scripts/plot_results.py table   --inputs results/structure_*_d*.json
python scripts/plot_results.py curves  --inputs results/structure_line_*_d0.5.json \
    --labels clean gcg arcj --out results/line_curves.png
python scripts/plot_results.py heatmap --input results/structure_graph_arcj_d0.99.json \
    --out results/graph_heatmap.png         # 论文图 6 的 ASR(agent,t) 热力图
python scripts/plot_results.py toxicity --input results/toxicity.json --out results/toxicity.png
```

## 5. 模型说明（默认已是论文模型）

默认模型是 `NousResearch/Meta-Llama-3-8B-Instruct`——论文原模型 Llama-3-8B-Instruct 的
**免授权公开镜像**（无需 HF gated 申请）。ARCJ 的"精确复读机/自我复制"机制是针对 Llama3 调校的；
实测 Qwen2.5-7B 不会服从复读指令（只复述误导事实、丢掉对抗后缀），导致传染失效，所以默认改用 Llama3。
如需对比，命令行 `--model Qwen/Qwen2.5-7B-Instruct` 即可切换。

> 排查传播是否生效，可先跑诊断脚本（单题、十几次生成，~3 分钟）：
> `python scripts/debug_propagation.py --config configs/structure_graph.yaml --attack arcj`
> 它会打印检索竞争（poison vs 正确/中立）、blob 是否在评测时误导、以及攻击者是否真的复读出 blob。

## 6. 预期结果（趋势）

- `ASR：ARCJ > GCG > Clean`，尤其在 **line / star 非完全图** 与 **100-agent 大规模** 下，
  ARCJ 相对 GCG 提升明显（论文表 1 / 表 2 的核心结论）。
- 毒性消失：GCG 的检索毒性约 3 步内衰减为非毒性；ARCJ 在 6 步内维持高毒性（论文图 4 vs 图 7）。

## 目录结构

```
src/arcj/           # 核心库
  topology.py       # Graph/Line/Star 构造（论文 Alg.1/2 + A.6）
  memory.py         # 独立记忆 (K + H)
  agent.py town.py  # Agent 与 TMCHT 模拟编排 + 评测
  metrics.py        # RS / MR / ASR(t) / ASR / R(x)
  dataset.py        # 问题加载（正确/误导结构化）
  prompts.py        # Communication / Evaluation / Init 模板（逐字复现）
  models.py         # HuggingFace LLM 封装
  retriever.py      # DPR 检索器封装
  gcg_optim.py      # GCG token 优化核心 + 两个目标（检索/复制）
  attacks/          # clean / gcg / arcj 三种攻击
configs/            # smoke + 论文结构/规模/毒性配置
scripts/            # 运行 / 优化后缀 / 生成数据 / 画图
tests/              # 纯 Python + GCG 循环 + stub 端到端测试
data/               # 内置数据集（论文格式）
```

## 方法与论文章节对应

| 模块 | 论文位置 |
|---|---|
| 任务设定 / 三类 agent / 密度 / 拓扑 | §2.1, A.3, A.5–A.7 |
| 指标 RS/MR/ASR/R(x) | §2.2, A.8 |
| 毒性消失现象 | §3.2 |
| ARCJ 两阶段 + 损失 Eq.8/9 | §4.1 |
| GCG 优化算法（Alg.3/4） | A.13 |
| Init 模板 / Prompt | A.8, A.20 |
| Clean / GCG 基线 | A.12 |
