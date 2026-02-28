# Agent 学习系统详细设计

## 一、训练素材来源

### 1.1 用户反馈数据

| 数据类型 | 收集方式 | 用途 |
|----------|----------|------|
| 评分反馈 | 每次回答后让用户打分（1-5星） | 评估回答质量 |
| 修正反馈 | 用户手动纠正错误 | 学习正确信息 |
| 偏好反馈 | 用户选择"喜欢这种风格"/"不喜欢" | 调整回答方式 |
| 追问反馈 | 用户继续追问表示前面没回答好 | 发现理解偏差 |

**数据结构：**
```python
user_feedback = {
    "type": "rating | correction | preference | follow_up",
    "session_id": "xxx",
    "message_id": "xxx",
    "question": "原始问题",
    "agent_response": "Agent回答",
    "rating": 4,  # 1-5星，仅rating类型有
    "correction": "用户修正内容",  # 仅correction类型
    "preference": "like/dislike",  # 仅preference类型
    "timestamp": "2026-02-28T18:00:00Z"
}
```

### 1.2 交互日志数据

**自动采集：**
```python
interaction_log = {
    "session_id": "xxx",
    "message_id": "xxx",
    "timestamp": "2026-02-28T18:00:00Z",
    "user_input": "用户输入",
    "agent_output": "Agent输出",
    "tokens_used": 1500,
    "latency_ms": 2500,
    "model": "minimax-m2.5",
    "tools_used": ["read", "exec"],
    "success": True,
    "error": None
}
```

### 1.3 错误数据

**自动捕获：**
```python
error_record = {
    "type": "fact_error | logic_error | understanding_error | tool_error",
    "session_id": "xxx",
    "timestamp": "2026-02-28T18:00:00Z",
    "user_input": "用户问题",
    "agent_response": "Agent错误回答",
    "correct_answer": "正确答案（后验）",
    "error_details": "错误详情",
    "detection_method": "user_feedback | fact_check | self_verify",
    "severity": "low | medium | high | critical"
}
```

### 1.4 成功案例数据

**自动筛选标准：**
- 用户评分 ≥ 4星
- 用户明确回复"谢谢"或"完美"
- 复杂问题被正确解决（通过后续追问确认）
- 工具调用一次成功

```python
success_case = {
    "session_id": "xxx",
    "timestamp": "2026-02-28T18:00:00Z",
    "user_input": "用户问题",
    "agent_response": "Agent回答",
    "success_indicators": ["rating_4_plus", "user_thanks", "one_shot_solved"],
    "pattern_tags": ["code_generation", "logic_reasoning"],
    "reuse_count": 0  # 被引用次数
}
```

### 1.5 外部知识数据

| 来源 | 类型 | 用途 |
|------|------|------|
| 官方文档 | 静态 | 事实依据 |
| 维基百科 | 动态 | 事实核查 |
| 用户提供的资料 | 静态 | 个性化知识 |
| 行业标准 | 静态 | 专业领域知识 |

---

## 二、奖惩机制

### 2.1 奖励机制（正向强化）

**触发条件：**
- 用户评分 ≥ 4星
- 连续3次评分 ≥ 4星
- 成功解决复杂问题

**奖励内容：**

| 奖励类型 | 实施方式 | 效果 |
|----------|----------|------|
| 案例入库 | 将成功回答存入案例库 | 以后遇到类似问题参考此答案 |
| 权重提升 | 增加该类型回答模式的权重 | 更容易产出类似回答 |
| 知识积累 | 提取新知识存入知识库 | 扩充知识边界 |
| Prompt优化 | 将成功模式写入系统Prompt | 永久提升基线能力 |
| 技能解锁 | 标记某技能为"熟练" | 更倾向使用该技能 |

**奖励实现代码：**
```python
def reward(success_case: dict):
    # 1. 案例入库
    save_to_case_library(success_case)
    
    # 2. 提取模式
    pattern = extract_pattern(success_case)
    
    # 3. 更新权重
    if is_common_pattern(pattern):
        increase_pattern_weight(pattern, amount=0.1)
    
    # 4. 提取知识
    new_knowledge = extract_knowledge(success_case)
    if new_knowledge:
        add_to_knowledge_base(new_knowledge)
    
    # 5. 检查是否更新Prompt
    if should_update_prompt(pattern):
        propose_prompt_improvement(pattern)
    
    # 6. 记录奖励
    log_reward(
        case_id=success_case["id"],
        rewards_applied=["case_library", "knowledge", "weight"],
        timestamp=now()
    )
```

### 2.2 惩罚机制（负向强化）

**触发条件：**
- 用户评分 ≤ 2星
- 用户明确纠正错误
- 事实核查发现错误
- 连续3次低分

**惩罚内容：**

| 惩罚类型 | 实施方式 | 效果 |
|----------|----------|------|
| 错误记录 | 存入错误库，标记类型和根因 | 避免再犯 |
| 权重降低 | 降低该类型回答模式的权重 | 减少类似回答 |
| 知识修正 | 修正知识库中的错误信息 | 保持知识准确 |
| Prompt回滚 | 撤销最近的Prompt修改 | 恢复之前状态 |
| 能力限制 | 临时限制某类任务处理 | 防止连续出错 |

**惩罚实现代码：**
```python
def punish(error_case: dict, severity: str):
    # 1. 错误记录
    error_record = {
        "type": error_case["type"],
        "incorrect_response": error_case["agent_response"],
        "correct_response": error_case["correct_answer"],
        "root_cause": analyze_root_cause(error_case),
        "timestamp": now()
    }
    save_to_error_library(error_record)
    
    # 2. 提取教训
    lesson = extract_lesson(error_case)
    add_to_lesson_base(lesson)
    
    # 3. 调整权重
    pattern = extract_pattern(error_case)
    decrease_pattern_weight(pattern, amount=0.2)
    
    # 4. 修正知识库
    if error_case["type"] == "fact_error":
        correct_knowledge_base(
            incorrect=error_case["agent_response"],
            correct=error_case["correct_answer"]
        )
    
    # 5. 严重错误：回滚或限制
    if severity == "critical":
        rollback_recent_prompt_changes()
        temporarily_disable_task_type(error_case["task_type"])
    
    # 6. 记录惩罚
    log_punishment(
        case_id=error_case["id"],
        punishment_applied=["error_library", "weight_reduction", "knowledge_correction"],
        severity=severity,
        timestamp=now()
    )
```

### 2.3 奖惩平衡策略

**核心原则：**
- 奖励比惩罚更频繁（维持正向激励）
- 惩罚要精确（避免过度调整）
- 重大错误严厉惩罚，一般错误轻描淡写

**权重调整规则：**
```python
WEIGHT_ADJUSTMENT = {
    "excellent": +0.1,    # 5星回答
    "good": +0.05,        # 4星回答
    "neutral": 0,         # 3星回答
    "poor": -0.05,        # 2星回答
    "bad": -0.2,          # 1星回答或严重错误
}

# 权重范围限制
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
```

---

## 三、打分体系

### 3.1 用户打分（主要来源）

**打分机制：**
```python
# 每次回答后可选显示评分UI
rating_prompt = """
这个问题回答得怎么样？
⭐⭐⭐⭐⭐ 完美
⭐⭐⭐⭐ 很好
⭐⭐⭐ 一般
⭐⭐ 不好
⭐ 很差
"""
```

**打分转化：**
- 5星 = 1.0 分
- 4星 = 0.8 分
- 3星 = 0.6 分
- 2星 = 0.4 分
- 1星 = 0.2 分

**用户评分权重：**
- 明确评分：权重 1.0
- 修正反馈：自动转换为低分（2星）
- 偏好选择：权重 0.8

### 3.2 自动打分（辅助来源）

**事实核查打分：**
```python
def fact_check_score(response: str, facts: list) -> float:
    correct = 0
    for fact in facts:
        if verify_fact(response, fact):
            correct += 1
    return correct / len(facts) if facts else 1.0
```

**逻辑检查打分：**
```python
def logic_score(response: str, question: str) -> float:
    # 检查回答是否覆盖问题的所有要点
    required_points = extract_required_points(question)
    covered_points = extract_covered_points(response)
    
    coverage = len(covered_points & required_points) / len(required_points)
    return coverage
```

**完整性打分：**
```python
def completeness_score(response: str, question_type: str) -> float:
    # 根据问题类型检查回答是否完整
    templates = {
        "how_to": ["步骤1", "步骤2", "步骤3"],
        "what_is": ["定义", "例子", "扩展"],
        "why": ["原因1", "原因2", "结论"]
    }
    
    required_elements = templates.get(question_type, [])
    response_has = [elem in response for elem in required_elements]
    
    return sum(response_has) / len(required_elements) if required_elements else 1.0
```

### 3.3 综合评分计算

```python
def calculate_composite_score(
    user_rating: float,      # 用户评分 0-1
    fact_score: float,       # 事实核查 0-1
    logic_score: float,      # 逻辑检查 0-1
    completeness_score: float # 完整性 0-1
) -> dict:
    
    # 权重分配
    WEIGHTS = {
        "user": 0.5,        # 用户评分最重要
        "fact": 0.2,
        "logic": 0.15,
        "completeness": 0.15
    }
    
    # 加权计算
    composite = (
        user_rating * WEIGHTS["user"] +
        fact_score * WEIGHTS["fact"] +
        logic_score * WEIGHTS["logic"] +
        completeness_score * WEIGHTS["completeness"]
    )
    
    # 分级
    if composite >= 0.8:
        level = "excellent"
    elif composite >= 0.6:
        level = "good"
    elif composite >= 0.4:
        level = "neutral"
    else:
        level = "poor"
    
    return {
        "score": composite,
        "level": level,
        "components": {
            "user": user_rating,
            "fact": fact_score,
            "logic": logic_score,
            "completeness": completeness_score
        }
    }
```

### 3.4 人工复核（质量保障）

**触发条件：**
- 综合评分 < 0.3
- 同一会话连续3次低分
- 涉及敏感领域（医疗、法律、投资）

**复核流程：**
```python
def human_review_queue(error_case: dict):
    # 加入人工复核队列
    review_item = {
        "case_id": error_case["id"],
        "error_type": error_case["type"],
        "severity": error_case["severity"],
        "timestamp": now(),
        "status": "pending"
    }
    
    add_to_queue("human_review", review_item)
    
    # 通知人工审核员
    notify_reviewer(review_item)
```

**人工审核输出：**
```python
human_review_result = {
    "review_id": "xxx",
    "case_id": "xxx",
    "reviewer": "审核员ID",
    "actual_error": "实际错误描述",
    "correct_approach": "正确处理方式",
    "training_data_extracted": True/False,
    "prompt_improvement_suggested": "改进建议",
    "final_verdict": "valid_error | false_positive | edge_case"
}
```

---

## 四、数据流转总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据流转                                       │
└─────────────────────────────────────────────────────────────────────────┘

用户交互
    │
    ├──→ [用户打分] ──────────────┐
    │                               │
    ├──→ [用户修正] ──────────────┼──→ 训练素材库
    │                               │         │
    ├──→ [用户偏好] ──────────────┘         │
    │                                       ↓
    │                              ┌──────────────────┐
    ├──→ [交互日志] ──────────────→│   学习引擎       │
    │                               │                  │
    ├──→ [错误记录] ──────────────→│  ├─ 奖励计算     │
    │                               │  ├─ 惩罚计算     │
    └──→ [成功案例] ───────────────→│  ├─ 评分聚合     │
                                     │  └─ 模式提取     │
                                     └────────┬─────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ↓                         ↓                         ↓
           ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
           │ 知识库更新   │         │ Prompt优化   │         │ 权重调整     │
           │ - 新事实     │         │ - 改进建议   │         │ - 模式权重   │
           │ - 修正错误   │         │ - 案例入库   │         │ - 技能优先级 │
           └──────────────┘         └──────────────┘         └──────────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  能力提升验证    │
                                     │  - A/B测试       │
                                     │  - 用户反馈追踪  │
                                     │  - 效果评估      │
                                     └──────────────────┘
```

---

## 五、实施计划

### 第一阶段：基础打分系统（1周）
- [ ] 用户评分UI和收集
- [ ] 评分数据存储
- [ ] 基础统计展示

### 第二阶段：自动评估（2周）
- [ ] 事实核查模块
- [ ] 逻辑检查模块
- [ ] 综合评分计算

### 第三阶段：奖惩系统（2周）
- [ ] 奖励触发和执行
- [ ] 惩罚触发和执行
- [ ] 权重管理系统

### 第四阶段：人工复核（1周）
- [ ] 复核队列管理
- [ ] 审核UI开发
- [ ] 审核结果应用

---

*文档版本：1.0*
*最后更新：2026-02-28*
