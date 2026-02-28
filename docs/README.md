# Agent 智能优化系统

## 简介

一个让 Agent 越用越聪明的完整方案，包含三大核心模块。

## 系统架构

```
用户输入 → 模型选用(Router) → 生成回答 → 自动评分(Scorer) → 自我学习(Learner) → 返回结果
```

## 三大模块

### 1. 模型路由器 (Router)
- 根据任务类型自动选择最优模型
- 支持：逻辑推理、代码生成、问答、总结、翻译等
- 文件：src/model_router.py

### 2. 自动评分器 (Scorer)
- 无需用户打分，自动评估回答质量
- 评分维度：自洽性、事实准确性、LLM评估、异常检测、风格匹配
- 文件：src/auto_scorer.py

### 3. 自我学习器 (Learner)
- 从成功/失败案例中学习
- 案例库、错误库、Prompt改进建议
- 文件：src/self_learner.py

## 快速开始

```bash
cd src
python model_router.py
python auto_scorer.py
python self_learner.py
python main.py
```

## 评分权重

- LLM评估: 30%
- 事实准确性: 25%
- 自洽性: 15%
- 异常检测: 15%
- 风格匹配: 15%

## 模型选用策略

- 逻辑推理: Llama 3.3 70B
- 代码生成: MiniMax M2.5
- 快速问答: Mixtral 8x7B

## 学习机制

- 高分(≥0.8) → 成功案例入库
- 低分(≤0.4) → 错误记录+教训提取
- 连续3次低分 → 触发Prompt优化建议

最后更新：2026-02-28