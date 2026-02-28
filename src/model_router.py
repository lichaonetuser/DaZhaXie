"""
模型路由器 - 根据任务类型自动选择最优模型

功能：
1. 任务分类：根据用户输入识别任务类型（逻辑推理、代码生成、问答、总结等）
2. 模型选择：根据任务类型选择最适合的模型
3. 性能追踪：记录各模型的历史表现，动态调整选择策略
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import json
from pathlib import Path


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
    name: str                           # 模型名称
    endpoint: str                       # API端点
    task_types: List[TaskType]          # 支持的任务类型
    speed_score: float = 0.5            # 速度评分 0-1，越高越快
    quality_score: float = 0.5          # 质量评分 0-1，越高越好
    cost_per_1k_tokens: float = 0.001   # 每1000 token成本
    
    @property
    def 综合评分(self) -> float:
        """综合评分 = 速度*0.3 + 质量*0.7"""
        return self.speed_score * 0.3 + self.quality_score * 0.7


class ModelRouter:
    """
    模型路由器
    
    根据用户输入自动分类任务类型，并选择最优模型。
    支持动态调整：基于历史表现调整模型选择权重。
    """
    
    # 模型注册表 - 可扩展
    MODELS: Dict[str, ModelInfo] = {
        "llama-3.3-70b": ModelInfo(
            name="Llama 3.3 70B",
            endpoint="meta/llama-3.3-70b-instruct",
            task_types=[TaskType.LOGIC_REASONING, TaskType.SUMMARIZATION, TaskType.CLASSIFICATION],
            speed_score=0.6,
            quality_score=0.95,
            cost_per_1k_tokens=0.001
        ),
        "minimax-m2.5": ModelInfo(
            name="MiniMax M2.5",
            endpoint="minimaxai/minimax-m2.5",
            task_types=[TaskType.CODE_GENERATION, TaskType.DEFAULT, TaskType.CREATIVE_WRITING],
            speed_score=0.4,
            quality_score=0.85,
            cost_per_1k_tokens=0.001
        ),
        "mixtral-8x7b": ModelInfo(
            name="Mixtral 8x7B",
            endpoint="mistralai/mixtral-8x7b-instruct-v0.1",
            task_types=[TaskType.QUESTION_ANSWER, TaskType.DEFAULT, TaskType.TRANSLATION],
            speed_score=0.9,
            quality_score=0.75,
            cost_per_1k_tokens=0.0007
        ),
    }
    
    def __init__(self, storage_path: str = "~/.openclaw/router"):
        """
        初始化路由器
        
        Args:
            storage_path: 历史表现数据存储路径
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 加载历史表现数据
        self.performance_history: Dict[str, List[float]] = self._load_history()
        
        # 任务分类关键词映射
        self.task_keywords = {
            TaskType.LOGIC_REASONING: ["逻辑", "推理", "计算", "解方程", "证明", "分析"],
            TaskType.CODE_GENERATION: ["代码", "写程序", "函数", "算法", "实现", "编程", "Python", "JavaScript"],
            TaskType.QUESTION_ANSWER: ["什么是", "为什么", "怎么样", "如何", "问题", "问答"],
            TaskType.SUMMARIZATION: ["总结", "概括", "摘要", "核心观点", "要点"],
            TaskType.CREATIVE_WRITING: ["写", "创作", "故事", "诗歌", "文章", "文案"],
            TaskType.TRANSLATION: ["翻译", "英文", "中文", "语言转换"],
            TaskType.CLASSIFICATION: ["分类", "判断", "属于", "类型"],
        }
    
    def _load_history(self) -> Dict[str, List[float]]:
        """加载历史表现数据"""
        history_file = self.storage_path / "performance.json"
        if history_file.exists():
            with open(history_file) as f:
                return json.load(f)
        return {}
    
    def _save_history(self):
        """保存历史表现数据"""
        history_file = self.storage_path / "performance.json"
        with open(history_file, "w") as f:
            json.dump(self.performance_history, f, ensure_ascii=False, indent=2)
    
    def classify_task(self, user_input: str) -> TaskType:
        """
        根据用户输入自动分类任务类型
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            TaskType: 识别出的任务类型
        """
        text = user_input.lower()
        
        # 匹配任务类型
        for task_type, keywords in self.task_keywords.items():
            if any(keyword in text for keyword in keywords):
                return task_type
        
        return TaskType.DEFAULT
    
    def select_model(
        self, 
        task_type: TaskType, 
        prefer_speed: bool = False,
        prefer_quality: bool = True
    ) -> ModelInfo:
        """
        根据任务类型选择最优模型
        
        Args:
            task_type: 任务类型
            prefer_speed: 是否优先考虑速度
            prefer_quality: 是否优先考虑质量
            
        Returns:
            ModelInfo: 选中的模型信息
        """
        # 筛选支持该任务类型的模型
        candidates = [
            m for m in self.MODELS.values() 
            if task_type in m.task_types or TaskType.DEFAULT in m.task_types
        ]
        
        if not candidates:
            # 如果没有匹配的，返回默认模型
            candidates = [self.MODELS["minimax-m2.5"]]
        
        # 根据偏好排序
        if prefer_speed:
            candidates.sort(key=lambda m: m.speed_score, reverse=True)
        elif prefer_quality:
            candidates.sort(key=lambda m: m.quality_score, reverse=True)
        else:
            candidates.sort(key=lambda m: m.综合评分, reverse=True)
        
        # 考虑历史表现：如果某模型历史评分低于0.5，降低优先级
        selected = candidates[0]
        for model in candidates:
            history = self.performance_history.get(model.name, [])
            if history:
                avg_score = sum(history) / len(history)
                if avg_score < 0.5:
                    continue  # 跳过历史表现差的模型
            selected = model
            break
        
        return selected
    
    def record_performance(self, model_name: str, score: float):
        """
        记录模型表现
        
        Args:
            model_name: 模型名称
            score: 评分 0-1
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(score)
        
        # 只保留最近20条记录
        self.performance_history[model_name] = self.performance_history[model_name][-20:]
        
        self._save_history()
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """根据名称获取模型信息"""
        return self.MODELS.get(name)
    
    def route(self, user_input: str, **kwargs) -> ModelInfo:
        """
        路由入口 - 自动选择模型
        
        Args:
            user_input: 用户输入
            **kwargs: 传递给select_model的参数
            
        Returns:
            ModelInfo: 选中的模型
        """
        task_type = self.classify_task(user_input)
        return self.select_model(task_type, **kwargs)
    
    def get_stats(self) -> Dict:
        """获取路由器统计信息"""
        return {
            "total_models": len(self.MODELS),
            "performance_records": {k: len(v) for k, v in self.performance_history.items()},
            "supported_task_types": [t.value for t in TaskType]
        }


# ==================== 测试用例 ====================

def test_model_router():
    """测试模型路由器"""
    print("=" * 50)
    print("测试：ModelRouter")
    print("=" * 50)
    
    router = ModelRouter()
    
    # 测试用例
    test_cases = [
        ("帮我解这道数学题：有3个红箱子，每个蓝箱子比红箱子多2个球，总球数27，求红蓝箱子各多少球？", TaskType.LOGIC_REASONING),
        ("写一个Python算法，找出数组中出现次数超过n/3的元素", TaskType.CODE_GENERATION),
        ("今天天气怎么样？", TaskType.QUESTION_ANSWER),
        ("总结一下这篇文章的主要内容", TaskType.SUMMARIZATION),
        ("写一首关于春天的诗", TaskType.CREATIVE_WRITING),
        ("把这段话翻译成英文", TaskType.TRANSLATION),
        ("这段代码属于什么类型？", TaskType.CLASSIFICATION),
        ("你好，请帮我个忙", TaskType.DEFAULT),
    ]
    
    passed = 0
    failed = 0
    
    for user_input, expected_type in test_cases:
        # 任务分类测试
        actual_type = router.classify_task(user_input)
        type_ok = actual_type == expected_type
        
        # 模型选择测试
        model = router.select_model(actual_type)
        
        status = "✓" if type_ok else "✗"
        print(f"\n{status} 输入: {user_input[:30]}...")
        print(f"  预期类型: {expected_type.value}, 实际: {actual_type.value}")
        print(f"  选用模型: {model.name} (质量:{model.quality_score}, 速度:{model.speed_score})")
        
        if type_ok:
            passed += 1
        else:
            failed += 1
    
    # 测试性能记录
    print("\n" + "-" * 50)
    router.record_performance("Llama 3.3 70B", 0.85)
    router.record_performance("Llama 3.3 70B", 0.90)
    router.record_performance("MiniMax M2.5", 0.75)
    print("已记录测试性能数据")
    
    # 测试统计
    print("\n" + "-" * 50)
    stats = router.get_stats()
    print(f"统计信息: {stats}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    test_model_router()