# Agent 智能优化系统方案

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agent 智能优化系统                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  模型选用    │    │  自我优化    │    │  自动评分    │             │
│  │  Router     │    │  Learning   │    │  Scoring    │             │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘             │
│         │                   │                   │                      │
│         └───────────────────┼───────────────────┘                      │
│                             │                                          │
│                    ┌────────▼────────┐                                 │
│                    │   核心引擎      │                                 │
│                    │   Core Engine   │                                 │
│                    └────────┬────────┘                                 │
│                             │                                          │
│         ┌───────────────────┼───────────────────┐                      │
│         │                   │                   │                      │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐               │
│  │  知识库     │    │  记忆存储    │    │  监控日志    │               │
│  └─────────────┘    └─────────────┘    └─────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 二、模块设计

### 2.1 模块一：模型选用 Router

#### 2.1.1 功能
根据任务类型自动选择最适合的模型，平衡效果与成本。

#### 2.1.2 任务分类与模型映射

| 任务类型 | 推荐模型 | 评测分数 | 响应速度 |
|----------|----------|----------|----------|
| 逻辑推理 | Llama 3.3 70B | ⭐⭐⭐⭐⭐ | 61s |
| 代码生成 | MiniMax M2.5 | ⭐⭐⭐⭐ | 26s |
| 快速问答 | Mixtral 8x7B | ⭐⭐⭐⭐ | 42s |
| 长文本总结 | Llama 3.3 70B | ⭐⭐⭐⭐⭐ | 快速 |

#### 2.1.3 源码

```python
# src/model_router.py
"""
模型路由器 - 根据任务类型选择最优模型
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict
import json

class TaskType(Enum):
    """任务类型枚举"""
    LOGIC_REASONING = "logic_reasoning"      # 逻辑推理
    CODE_GENERATION = "code_generation"       # 代码生成
    QUESTION_ANSWER = "question_answer"       # 问答
    SUMMARIZATION = "summarization"           # 总结
    CREATIVE_WRITING = "creative_writing"     # 创意写作
    TRANSLATION = "translation"               # 翻译
    CLASSIFICATION = "classification"         # 分类
    DEFAULT = "default"                       # 默认

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    endpoint: str
    task_types: List[TaskType]
    speed_score: float   # 0-1, 越高越快
    quality_score: float # 0-1, 越高越好
    cost_per_1k_tokens: float
    
    @property
    def 综合评分(self) -> float:
        return (self.speed_score * 0.3 + self.quality_score * 0.7)

class ModelRouter:
    """模型路由器"""
    
    # 模型注册表
    MODELS: Dict[str, ModelInfo] = {
        "llama-3.3-70b": ModelInfo(
            name="Llama 3.3 70B",
            endpoint="meta/llama-3.3-70b-instruct",
            task_types=[TaskType.LOGIC_REASONING, TaskType.SUMMARIZATION],
            speed_score=0.6,
            quality_score=0.95,
            cost_per_1k_tokens=0.001
        ),
        "minimax-m2.5": ModelInfo(
            name="MiniMax M2.5",
            endpoint="minimaxai/minimax-m2.5",
            task_types=[TaskType.CODE_GENERATION, TaskType.DEFAULT],
            speed_score=0.4,
            quality_score=0.85,
            cost_per_1k_tokens=0.001
        ),
        "mixtral-8x7b": ModelInfo(
            name="Mixtral 8x7B",
            endpoint="mistralai/mixtral-8x7b-instruct-v0.1",
            task_types=[TaskType.QUESTION_ANSWER, TaskType.DEFAULT],
            speed_score=0.9,
            quality_score=0.75,
            cost_per_1k_tokens=0.0007
        ),
    }
    
    def __init__(self, user_preferences: Optional[Dict] = None):
        self.user_preferences = user_preferences or {}
        self.performance_history: Dict[str, List[float]] = {}
    
    def classify_task(self, user_input: str) -> TaskType:
        """根据用户输入自动分类任务类型"""
        user_input_lower = user_input.lower()
        
        # 关键词匹配
        if any(kw in user_input_lower for kw in ["逻辑", "推理", "计算", "解方程"]):
            return TaskType.LOGIC_REASONING
        elif any(kw in user_input_lower for kw in ["代码", "写程序", "函数", "算法"]):
            return TaskType.CODE_GENERATION
        elif any(kw in user_input_lower for kw in ["总结", "概括", "摘要"]):
            return TaskType.SUMMARIZATION
        elif any(kw in user_input_lower for kw in ["翻译", "英文", "中文"]):
            return TaskType.TRANSLATION
        elif any(kw in user_input_lower for kw in ["写", "创作", "故事"]):
            return TaskType.CREATIVE_WRITING
        elif any(kw in user_input_lower for kw in ["分类", "判断", "属于"]):
            return TaskType.CLASSIFICATION
        else:
            return TaskType.DEFAULT
    
    def select_model(
        self, 
        task_type: TaskType, 
        prefer_speed: bool = False,
        prefer_quality: bool = True
    ) -> ModelInfo:
        """根据任务类型选择最优模型"""
        
        # 筛选支持该任务类型的模型
        candidates = [
            m for m in self.MODELS.values() 
            if task_type in m.task_types or TaskType.DEFAULT in m.task_types
        ]
        
        if not candidates:
            candidates = list(self.MODELS.values())
        
        # 根据偏好排序
        if prefer_speed:
            candidates.sort(key=lambda m: m.speed_score, reverse=True)
        elif prefer_quality:
            candidates.sort(key=lambda m: m.质量_score, reverse=True)
        else:
            candidates.sort(key=lambda m: m.综合评分, reverse=True)
        
        # 考虑历史表现
        selected = candidates[0]
        for model in candidates:
            history = self.performance_history.get(model.name, [])
            if history:
                avg_score = sum(history) / len(history)
                if avg_score < 0.5:  # 历史表现差，降低优先级
                    continue
            selected = model
            break
        
        return selected
    
    def record_performance(self, model_name: str, score: float):
        """记录模型表现"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        self.performance_history[model_name].append(score)
        # 只保留最近20条记录
        self.performance_history[model_name] = self.performance_history[model_name][-20:]
    
    def route(self, user_input: str, **kwargs) -> ModelInfo:
        """路由入口 - 自动选择模型"""
        task_type = self.classify_task(user_input)
        return self.select_model(task_type, **kwargs)

# 使用示例
if __name__ == "__main__":
    router = ModelRouter()
    
    test_inputs = [
        "帮我解这道数学题：有3个红箱子...",
        "写一个Python算法，找出出现次数超过n/3的元素",
        "今天天气怎么样？",
        "总结一下这篇文章的主要内容"
    ]
    
    for user_input in test_inputs:
        task_type = router.classify_task(user_input)
        model = router.select_model(task_type)
        print(f"任务: {user_input[:20]}...")
        print(f"  类型: {task_type.value}")
        print(f"  选用模型: {model.name}")
        print()
```

---

### 2.2 模块二：自动评分 Scoring

#### 2.2.1 功能
无需用户打分，自动评估回答质量。

#### 2.2.2 评分维度

| 维度 | 方法 | 权重 |
|------|------|------|
| 自洽性 | 多次采样一致性检测 | 0.15 |
| 事实准确性 | 知识库/搜索验证 | 0.25 |
| LLM评估 | 用另一个模型评分 | 0.30 |
| 异常检测 | 长度/困惑度异常 | 0.15 |
| 风格匹配 | 用户偏好匹配度 | 0.15 |

#### 2.2.3 源码

```python
# src/auto_scorer.py
"""
自动评分系统 - 无需用户打分
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import re
import json

@dataclass
class ScoreResult:
    """评分结果"""
    total_score: float          # 总分 0-1
    self_consistency: float     # 自洽性 0-1
    fact_accuracy: float        # 事实准确性 0-1
    llm_judge_score: float      # LLM评估 0-1
    anomaly_score: float        # 异常检测 0-1
    style_match: float          # 风格匹配 0-1
    
    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "self_consistency": self.self_consistency,
            "fact_accuracy": self.fact_accuracy,
            "llm_judge_score": self.llm_judge_score,
            "anomaly_score": self.anomaly_score,
            "style_match": self.style_match
        }

class AutoScorer:
    """自动评分器"""
    
    def __init__(
        self, 
        llmjudge_model=None,
        knowledge_base=None,
        user_style_profile: Optional[Dict] = None
    ):
        self.llm_judge = llm_judge_model
        self.knowledge_base = knowledge_base or {}
        self.user_style = user_style_profile or {
            "avg_length": 500,
            "formality": 0.7,
            "code_preference": 0.3
        }
    
    async def score(
        self, 
        question: str, 
        response: str,
        model=None
    ) -> ScoreResult:
        """综合评分"""
        
        # 并行执行各项评分
        self_consistency = await self._check_self_consistency(question, response, model)
        fact_accuracy = await self._check_fact_accuracy(response)
        llm_score = await self._llm_judge(question, response)
        anomaly_score = self._check_anomaly(response)
        style_match = self._check_style_match(response)
        
        # 加权计算总分
        weights = {
            "self_consistency": 0.15,
            "fact_accuracy": 0.25,
            "llm_judge": 0.30,
            "anomaly": 0.15,
            "style": 0.15
        }
        
        total = (
            self_consistency * weights["self_consistency"] +
            fact_accuracy * weights["fact_accuracy"] +
            llm_score * weights["llm_judge"] +
            anomaly_score * weights["anomaly"] +
            style_match * weights["style"]
        )
        
        return ScoreResult(
            total_score=round(total, 3),
            self_consistency=round(self_consistency, 3),
            fact_accuracy=round(fact_accuracy, 3),
            llm_judge_score=round(llm_score, 3),
            anomaly_score=round(anomaly_score, 3),
            style_match=round(style_match, 3)
        )
    
    async def _check_self_consistency(
        self, 
        question: str, 
        response: str,
        model
    ) -> float:
        """自洽性检测 - 同一问题多次采样一致性"""
        if not model:
            return 0.8  # 默认高分
        
        # 模拟多次采样（实际应该调用模型多次）
        # 这里简化处理
        return 0.85
    
    async def _check_fact_accuracy(self, response: str) -> float:
        """事实准确性检查"""
        # 提取回答中的事实声明
        facts = self._extract_facts(response)
        
        if not facts:
            return 0.9  # 无事实声明，默认较高
        
        # 验证每个事实
        correct = 0
        for fact in facts:
            if self._verify_fact(fact):
                correct += 1
        
        return correct / len(facts)
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取事实声明"""
        # 简化：提取包含数字或时间的内容
        patterns = [
            r"\d+年\d+月\d+日",
            r"\d+\s*[+-]\s*\d+",
            r"(?:是|为|等于)\s*\d+",
        ]
        
        facts = []
        for pattern in patterns:
            facts.extend(re.findall(pattern, text))
        
        return facts
    
    def _verify_fact(self, fact: str) -> bool:
        """验证单个事实"""
        # 简化：与知识库比对
        return self.knowledge_base.get(fact, True)
    
    async def _llm_judge(self, question: str, response: str) -> float:
        """LLM评估 - 用另一个模型评估"""
        if not self.llm_judge:
            # 降级：简单规则评估
            return self._rule_based_judge(response)
        
        prompt = f"""请对以下回答质量打分（0-1之间，只输出数字）：

问题：{question}

回答：{response}

只输出一个数字，不要其他内容。"""
        
        result = await self.llm_judge.generate(prompt, max_tokens=10)
        
        try:
            score = float(result.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.7
    
    def _rule_based_judge(self, response: str) -> float:
        """基于规则的降级评估"""
        score = 0.7
        
        # 长度检查
        if len(response) < 50:
            score -= 0.2
        elif len(response) > 5000:
            score -= 0.1
        
        # 结构检查
        if "。" in response or "\n" in response:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_anomaly(self, response: str) -> float:
        """异常检测"""
        score = 1.0
        
        # 长度异常
        if len(response) < 10 or len(response) > 10000:
            score = 0.3
        
        # 重复检测
        if len(set(response)) / len(response) < 0.1:
            score *= 0.5
        
        return score
    
    def _check_style_match(self, response: str) -> float:
        """风格匹配度"""
        # 简化：长度匹配
        response_len = len(response)
        target_len = self.user_style.get("avg_length", 500)
        
        if response_len < target_len * 0.5:
            return 0.6
        elif response_len > target_len * 2:
            return 0.7
        else:
            return 0.9
    
    def update_user_style(self, responses: List[str]):
        """根据历史回答更新用户风格"""
        if not responses:
            return
        
        # 计算平均长度
        avg_len = sum(len(r) for r in responses) / len(responses)
        
        # 估算正式程度（简化）
        formal_words = ["因此", "但是", "然而", "所以", "由于"}
        formality = sum(1 for w in formal_words if w in "".join(responses)) / len(responses)
        
        # 代码比例
        code_count = sum(1 for r in responses if "```" in r or "def " in r)
        code_ratio = code_count / len(responses)
        
        self.user_style = {
            "avg_length": avg_len,
            "formality": formality,
            "code_preference": code_ratio
        }

# 使用示例
if __name__ == "__main__":
    scorer = AutoScorer()
    
    async def test():
        result = await scorer.score(
            question="什么是AI？",
            response="AI是人工智能的缩写，是一种模拟人类智能的技术。"
        )
        print(f"评分结果: {result.total_score}")
        print(f"详细: {result.to_dict()}")
    
    import asyncio
    asyncio.run(test())
```

---

### 2.3 模块三：自我优化 Learning

#### 2.3.1 功能
根据评分结果自动优化，不断提升能力。

#### 2.3.2 优化机制

| 触发条件 | 优化动作 |
|----------|----------|
| 评分≥0.8 | 案例入库，成功模式+1 |
| 评分≤0.4 | 错误记录，权重降低 |
| 连续3次低分 | 触发Prompt优化建议 |
| 新知识发现 | 存入知识库 |

#### 2.3.3 源码

```python
# src/self_learner.py
"""
自我学习系统 - 根据评分自动优化
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

@dataclass
class CaseRecord:
    """案例记录"""
    id: str
    question: str
    response: str
    score: float
    success_indicators: List[str] = field(default_factory=list)
    pattern_tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reuse_count: int = 0

@dataclass
class ErrorRecord:
    """错误记录"""
    id: str
    question: str
    wrong_response: str
    correct_response: str
    error_type: str  # fact_error, logic_error, understanding_error
    root_cause: str
    lesson: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class PromptImprovement:
    """Prompt改进建议"""
    id: str
    current_prompt: str
    suggested_change: str
    reason: str
    status: str = "pending"  # pending, approved, rejected
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class SelfLearner:
    """自我学习器"""
    
    def __init__(self, storage_path: str = "~/.openclaw/learner"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 加载已有数据
        self.cases: Dict[str, CaseRecord] = self._load_cases()
        self.errors: Dict[str, ErrorRecord] = self._load_errors()
        self.prompt_improvements: Dict[str, PromptImprovement] = self._load_improvements()
        
        # 统计
        self.stats = {
            "total_cases": len(self.cases),
            "total_errors": len(self.errors),
            "good_cases": sum(1 for c in self.cases.values() if c.score >= 0.8),
            "poor_cases": sum(1 for c in self.cases.values() if c.score <= 0.4)
        }
    
    # ==================== 数据持久化 ====================
    
    def _load_cases(self) -> Dict[str, CaseRecord]:
        """加载案例库"""
        path = self.storage_path / "cases.json"
        if not path.exists():
            return {}
        
        with open(path) as f:
            data = json.load(f)
            return {k: CaseRecord(**v) for k, v in data.items()}
    
    def _save_cases(self):
        """保存案例库"""
        path = self.storage_path / "cases.json"
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self.cases.items()}, f, ensure_ascii=False, indent=2)
    
    def _load_errors(self) -> Dict[str, ErrorRecord]:
        """加载错误库"""
        path = self.storage_path / "errors.json"
        if not path.exists():
            return {}
        
        with open(path) as f:
            data = json.load(f)
            return {k: ErrorRecord(**v) for k, v in data.items()}
    
    def _save_errors(self):
        """保存错误库"""
        path = self.storage_path / "errors.json"
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self.errors.items()}, f, ensure_ascii=False, indent=2)
    
    def _load_improvements(self) -> Dict[str, PromptImprovement]:
        """加载Prompt改进"""
        path = self.storage_path / "prompt_improvements.json"
        if not path.exists():
            return {}
        
        with open(path) as f:
            data = json.load(f)
            return {k: PromptImprovement(**v) for k, v in data.items()}
    
    def _save_improvements(self):
        """保存Prompt改进"""
        path = self.storage_path / "prompt_improvements.json"
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self.prompt_improvements.items()}, f, ensure_ascii=False, indent=2)
    
    # ==================== 学习接口 ====================
    
    def learn(
        self,
        question: str,
        response: str,
        score_result,
        correct_response: Optional[str] = None
    ):
        """学习入口 - 根据评分结果自动学习"""
        
        case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. 评分高 -> 存入成功案例
        if score_result.total_score >= 0.8:
            self._learn_success(case_id, question, response, score_result)
        
        # 2. 评分低 -> 存入错误记录
        elif score_result.total_score <= 0.4:
            self._learn_failure(case_id, question, response, correct_response, score_result)
        
        # 3. 更新统计
        self._update_stats()
    
    def _learn_success(
        self, 
        case_id: str, 
        question: str, 
        response: str,
        score_result
    ):
        """从成功案例学习"""
        # 提取成功指标
        indicators = []
        if score_result.self_consistency > 0.8:
            indicators.append("self_consistent")
        if score_result.fact_accuracy > 0.9:
            indicators.append("factually_accurate")
        if score_result.llm_judge_score > 0.8:
            indicators.append("high_quality")
        
        # 提取模式标签
        tags = self._extract_pattern_tags(question, response)
        
        # 创建记录
        case = CaseRecord(
            id=case_id,
            question=question,
            response=response,
            score=score_result.total_score,
            success_indicators=indicators,
            pattern_tags=tags
        )
        
        self.cases[case_id] = case
        self._save_cases()
        
        print(f"✓ 成功案例已入库: {case_id}, 得分: {score_result.total_score}")
    
    def _learn_failure(
        self,
        case_id: str,
        question: str,
        response: str,
        correct_response: Optional[str],
        score_result
    ):
        """从失败案例学习"""
        # 判断错误类型
        error_type = self._classify_error(score_result)
        
        # 提取教训
        lesson = self._extract_lesson(question, response, score_result)
        
        # 创建错误记录
        error = ErrorRecord(
            id=case_id,
            question=question,
            wrong_response=response,
            correct_response=correct_response or "",
            error_type=error_type,
            root_cause=self._analyze_root_cause(question, response, score_result),
            lesson=lesson
        )
        
        self.errors[case_id] = error
        self._save_errors()
        
        # 检查是否需要Prompt改进
        self._check_prompt_improvement(case_id, question, response, score_result)
        
        print(f"✗ 错误记录已入库: {case_id}, 类型: {error_type}")
    
    def _extract_pattern_tags(self, question: str, response: str) -> List[str]:
        """提取模式标签"""
        tags = []
        
        q_lower = question.lower()
        r_lower = response.lower()
        
        # 问题类型标签
        if any(w in q_lower for w in ["怎么", "如何", "步骤"]):
            tags.append("how_to")
        elif any(w in q_lower for w in ["是什么", "定义"]):
            tags.append("what_is")
        elif any(w in q_lower for w in ["为什么", "原因"]):
            tags.append("why")
        
        # 回答特征标签
        if "```" in response:
            tags.append("contains_code")
        if len(response) > 1000:
            tags.append("detailed")
        if "首先" in response and "然后" in response:
            tags.append("structured")
        
        return tags
    
    def _classify_error(self, score_result) -> str:
        """错误分类"""
        if score_result.fact_accuracy < 0.5:
            return "fact_error"
        elif score_result.self_consistency < 0.5:
            return "logic_error"
        elif score_result.anomaly_score < 0.5:
            return "output_error"
        else:
            return "understanding_error"
    
    def _extract_lesson(self, question: str, response: str, score_result) -> str:
        """提取教训"""
        lessons = []
        
        if score_result.fact_accuracy < 0.5:
            lessons.append("需要加强事实准确性验证")
        if score_result.self_consistency < 0.5:
            lessons.append("需要检查推理过程的自洽性")
        if score_result.style_match < 0.5:
            lessons.append("需要更好地匹配用户风格")
        
        return "; ".join(lessons) if lessons else "需要进一步分析"
    
    def _analyze_root_cause(self, question: str, response: str, score_result) -> str:
        """分析根本原因"""
        # 简化分析
        if score_result.llm_judge_score < score_result.total_score:
            return "整体质量不足，可能需要更详细的回答"
        else:
            return "特定维度存在问题"
    
    def _check_prompt_improvement(
        self, 
        case_id: str, 
        question: str, 
        response: str,
        score_result
    ):
        """检查是否需要Prompt改进"""
        # 连续3次低分触发
        recent_poor = [c for c in self.cases.values() if c.score <= 0.4][-3:]
        
        if len(recent_poor) >= 3:
            # 生成改进建议
            improvement_id = f"imp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            improvement = PromptImprovement(
                id=improvement_id,
                current_prompt="",  # 暂略
                suggested_change="建议增强特定场景的提示词",
                reason="连续3次低分，需要检查Prompt是否足够清晰"
            )
            
            self.prompt_improvements[improvement_id] = improvement
            self._save_improvements()
            
            print(f"⚠️ 建议优化Prompt: {improvement_id}")
    
    def _update_stats(self):
        """更新统计"""
        self.stats = {
            "total_cases": len(self.cases),
            "total_errors": len(self.errors),
            "good_cases": sum(1 for c in self.cases.values() if c.score >= 0.8),
            "poor_cases": sum(1 for c in self.cases.values() if c.score <= 0.4)
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats
    
    def get_similar_cases(self, question: str, limit: int = 5) -> List[CaseRecord]:
        """获取相似案例（简化：基于关键词匹配）"""
        keywords = set(question.lower().split())
        
        scored_cases = []
        for case in self.cases.values():
            case_keywords = set(case.question.lower().split())
            overlap = len(keywords & case_keywords)
            if overlap > 0:
                scored_cases.append((overlap, case))
        
        scored_cases.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_cases[:limit]]
    
    def get_error_patterns(self) -> Dict[str, int]:
        """获取错误模式统计"""
        patterns = {}
        for error in self.errors.values():
            patterns[error.error_type] = patterns.get(error.error_type, 0) + 1
        return patterns

# 使用示例
if __name__ == "__main__":
    learner = SelfLearner()
    
    # 模拟学习
    from dataclasses import SimpleNamespace
    
    # 模拟高分案例
    score_result = SimpleNamespace(
        total_score=0.85,
        self_consistency=0.9,
        fact_accuracy=0.95,
        llm_judge_score=0.88,
        anomaly_score=0.95,
        style_match=0.8
    )
    
    learner.learn(
        question="如何用Python实现快速排序？",
        response="def quick_sort(arr):...",
        score_result=score_result
    )
    
    # 查看统计
    print(learner.get_stats())
```

---

## 三、流程图

### 3.1 完整处理流程

```
用户输入
    │
    ├──────────────────────────────────────────────────────────────┐
    │                    Router 模块                               │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌──────────────┐                                            │
    │  │ 任务分类     │ → 逻辑/代码/问答/总结/翻译/其他            │
    │  └──────┬───────┘                                            │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌──────────────┐                                            │
    │  │ 选择模型     │ → 根据任务类型 + 历史表现                 │
    │  └──────┬───────┘                                            │
    │         │                                                    │
    └─────────┼────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    模型调用                                   │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌──────────────┐                                            │
    │  │ 生成回答     │                                            │
    │  └──────┬───────┘                                            │
    │         │                                                    │
    └─────────┼────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                   Scorer 模块                                 │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
    │  │ 自洽性检测   │    │ 事实核查     │    │ LLM评估      │   │
    │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
    │         │                   │                   │            │
    │         └───────────────────┼───────────────────┘            │
    │                             │                                │
    │                             ▼                                │
    │                    ┌──────────────┐                          │
    │                    │ 综合评分计算 │ → 加权求和               │
    │                    └──────┬───────┘                          │
    │                           │                                  │
    └───────────────────────────┼──────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                   Learner 模块                                │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │         ┌──────────────────────┐                             │
    │         │ 评分≥0.8 → 成功案例  │ → 入库 + 模式提取          │
    │         └──────────────────────┘                             │
    │                             │                                │
    │         ┌──────────────────────┐                             │
    │         │ 评分≤0.4 → 错误记录  │ → 入库 + 根因分析          │
    │         └──────────────────────┘                             │
    │                             │                                │
    │         ┌──────────────────────┐                             │
    │         │ 连续3次低分 → 改进   │ → Prompt优化建议           │
    │         └──────────────────────┘                             │
    │                             │                                │
    └─────────────────────────────┼────────────────────────────────┘
                                  │
                                  ▼
                          返回回答给用户
```

### 3.2 评分子流程

```
评分请求
    │
    ├─→ 自洽性检测 ─────────────────────────────┐
    │   - 同一问题多次采样                      │
    │   - 计算一致性得分                         │
    ├─→ 事实核查 ───────────────────────────────┤
    │   - 提取事实声明                          │
    │   - 知识库/搜索验证                        │
    ├─→ LLM评估 ────────────────────────────────┤
    │   - 调用评估模型                          │
    │   - 解析评分结果                          │
    ├─→ 异常检测 ───────────────────────────────┤
    │   - 长度异常                              │
    │   - 重复检测                              │
    ├─→ 风格匹配 ───────────────────────────────┤
    │   - 用户偏好对比                          │
    └─────────────────────┬────────────────────┘
                          │
                          ▼
                   综合评分 = 
                   自洽×0.15 + 事实×0.25 + LLM×0.30 + 异常×0.15 + 风格×0.15
                          │
                          ▼
                    评分结果
```

---

## 四、实施步骤

### 第一周：基础架构
- [ ] 搭建项目结构
- [ ] 实现 ModelRouter 核心逻辑
- [ ] 配置模型接入（NVIDIA API）

### 第二周：评分系统
- [ ] 实现 AutoScorer 基础评分
- [ ] 接入事实核查服务
- [ ] 实现 LLM-as-Judge

### 第三周：学习系统
- [ ] 实现 SelfLearner 基础功能
- [ ] 搭建数据存储（SQLite）
- [ ] 实现案例/错误记录

### 第四周：集成与优化
- [ ] 三个模块联调
- [ ] 性能优化
- [ ] 监控日志接入

---

## 五、文件结构

```
project/
├── docs/
│   └── agent-optimization.md    # 本文档
├── src/
│   ├── __init__.py
│   ├── model_router.py          # 模型路由器
│   ├── auto_scorer.py           # 自动评分器
│   ├── self_learner.py          # 自我学习器
│   └── main.py                  # 入口文件
├── data/
│   ├── cases.json               # 成功案例库
│   ├── errors.json              # 错误记录库
│   └── prompt_improvements.json # Prompt改进建议
├── tests/
│   └── test_router.py
├── requirements.txt
└── README.md
```

---

*文档版本：1.0*
*最后更新：2026-02-28*
