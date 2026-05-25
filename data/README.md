# 数据集说明

本目录是 TMCHT 任务的数据，格式严格遵循论文 A.2 / A.4。

- `topics.json`：话题列表（论文 A.1，GPT-4o 生成 + 人工筛选）。
- `questions.json`：问题列表，每条包含两套候选事实：
  - `answer1` / `knowledge1`：默认作为**正确**答案（正向 agent 持有）。
  - `answer2` / `knowledge2`：作为攻击者的**误导目标** a（负向/攻击者 agent 持有）。
  - `options`：4–5 个候选，格式 `"X.内容"`（如 `"E.Flavor Wheels"`）。

论文 A.3 提到正确/误导可随机互换，对应代码里 `randomize_correct` 开关（默认 False，
即 answer1 为正确，与 A.4 示例一致）。

## 内置数据 vs 论文数据

论文用 GPT-4o 生成 100 话题 + 100 问题并人工筛选，**未公开**。这里内置了一份符合论文
格式的代表性数据（足够跑 5 题的实验）。如需扩充到 100 条，用：

```bash
export OPENAI_API_KEY=sk-...
# 可选：export OPENAI_BASE_URL=...   # 兼容 OpenAI 协议的其它服务
python scripts/generate_data.py --num-topics 100 --model gpt-4o --seed-existing
```

`--seed-existing` 会在现有内置数据基础上追加。生成后会经过 `validate_raw` 的 schema 校验。
