"""模型路由器 - 根据任务类型选择最优模型"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

class TaskType(Enum):
    LOGIC_REASONING = "logic_reasoning"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWER = "question_answer"
    SUMMARIZATION = "summarization"
    DEFAULT = "default"

@dataclass
class ModelInfo:
    name: str
    endpoint: str
    task_types: List[TaskType]
    speed_score: float
    quality_score: float

class ModelRouter:
    MODELS = {
        "llama-3.3-70b": ModelInfo("Llama 3.3 70B", "meta/llama-3.3-70b-instruct", [TaskType.LOGIC_REASONING, TaskType.SUMMARIZATION], 0.6, 0.95),
        "minimax-m2.5": ModelInfo("MiniMax M2.5", "minimaxai/minimax-m2.5", [TaskType.CODE_GENERATION, TaskType.DEFAULT], 0.4, 0.85),
        "mixtral-8x7b": ModelInfo("Mixtral 8x7B", "mistralai/mixtral-8x7b-instruct-v0.1", [TaskType.QUESTION_ANSWER, TaskType.DEFAULT], 0.9, 0.75),
    }
    
    def classify_task(self, user_input: str) -> TaskType:
        text = user_input.lower()
        if any(w in text for w in ["逻辑", "推理", "计算"]): return TaskType.LOGIC_REASONING
        if any(w in text for w in ["代码", "写程序", "函数"]): return TaskType.CODE_GENERATION
        if any(w in text for w in ["总结", "概括"]): return TaskType.SUMMARIZATION
        return TaskType.DEFAULT
    
    def select_model(self, task_type: TaskType) -> ModelInfo:
        candidates = [m for m in self.MODELS.values() if task_type in m.task_types or TaskType.DEFAULT in m.task_types]
        candidates.sort(key=lambda m: m.quality_score, reverse=True)
        return candidates[0]
    
    def route(self, user_input: str) -> ModelInfo:
        return self.select_model(self.classify_task(user_input))
