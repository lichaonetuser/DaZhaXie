# Agent 自动打分系统技术方案

## 1. 技术概览

减少对用户打分的依赖，通过AI技术自动评估回答质量。

| 技术 | 作用 | 类型 |
|------|------|------|
| 自洽性检测 | 检查回答内部是否一致 | 无监督 |
| 事实核查 | 验证回答中的事实准确性 | 知识库/API |
| LLM-as-Judge | 用另一个模型评估 | 有监督 |
| 异常检测 | 识别异常/低质量回答 | 无监督 |
| 风格分析 | 检查回答风格是否符合偏好 | 统计 |
| 困惑度检测 | 检测回答是否流畅 | 无监督 |

---

## 2. 自洽性检测

### 2.1 多次采样一致性

同一问题用不同温度采样多次，高一致性=高分数。

```python
def self_consistency_score(question, model):
    responses = [model.generate(question, temperature=0.7+i*0.1) for i in range(3)]
    embeddings = encoder.encode(responses)
    avg_sim = sum(cosine(embeddings[i], embeddings[j]) for i,j in pairs) / len(pairs)
    return avg_sim
```

### 2.2 观点自洽

```python
def viewpoint_consistency(response, model):
    points = model.extract_key_points(response)
    contradictions = sum(1 for i,p1 in enumerate(points) for p2 in points[i+1:] if model.check_contradiction(p1,p2))
    return 1.0 - contradictions/max(len(points),1)
```

---

## 3. 事实核查

### 3.1 知识库验证
```python
def knowledge_base_check(response, kb):
    facts = extract_facts(response)
    correct = sum(1 for f in facts if kb.verify(f))
    return correct/len(facts) if facts else 1.0
```

### 3.2 搜索验证
```python
def search_verification(response, search_api):
    claims = extract_claims(response)
    correct = sum(1 for c in claims if search_api.verify(c))
    return correct/len(claims) if claims else 1.0
```

---

## 4. LLM-as-Judge

### 4.1 评估Prompt
```python
EVAL_PROMPT = """评分：问题={question}，回答={response}
从准确性、完整性、清晰度、有用性打分(1-10)，输出JSON"""
```

### 4.2 实现
```python
def llm_judge(question, response, judge_model):
    result = judge_model.generate(EVAL_PROMPT.format(question=question, response=response))
    return json.loads(result)
```

### 4.3 自我评估
```python
def self_judge(question, response, model):
    score = float(model.generate(f"给这个回答打分0-1：{response}", max_tokens=5))
    return min(max(score, 0), 1)
```

---

## 5. 异常检测

### 5.1 长度异常
```python
def length_anomaly(response, history):
    mean, std = statistics.mean(history), statistics.stdev(history)
    z = abs(len(response)-mean)/(std or 1)
    return 0.0 if z > 3 else 1.0
```

### 5.2 质量异常（困惑度）
```python
def quality_anomaly(response, model):
    ppl = model.perplexity(response)
    return 0.0 if ppl>100 else 0.5 if ppl>50 else 1.0
```

---

## 6. 风格匹配

### 6.1 学习用户偏好
```python
def learn_user_style(responses):
    return {"avg_len":mean(map(len,responses)), 
            "formality":estimate_formality(responses),
            "code_ratio":count_code(responses)/len(responses)}
```

### 6.2 匹配度
```python
def style_match(response, pref):
    current = analyze_style(response)
    return mean(1-abs(current.get(k,0)-pref.get(k,0)) for k in pref)
```

---

## 7. 综合评分

| 维度 | 权重 |
|------|------|
| LLM评估 | 0.30 |
| 事实核查 | 0.25 |
| 自洽性 | 0.15 |
| 异常检测 | 0.15 |
| 风格匹配 | 0.15 |

流程：回答 → 自洽检测 → 事实核查 → LLM评估 → 异常检测 → 风格匹配 → 加权求和 → 最终分数

---

## 8. 与用户打分互补

| 场景 | 自动:用户 |
|------|----------|
| 新用户 | 0.8:0.2 |
| 有历史 | 0.5:0.5 |
| 样本足 | 0.2:0.8 |

用用户评分数据训练线性回归校准自动评分。

---

## 9. 实施计划

- 第1周：自洽性 + 异常检测
- 第2周：LLM评估集成  
- 第3周：事实核查 + 风格匹配
- 第4周：校准优化

---

*版本：1.0 更新：2026-02-28*
