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
    MATHEMATICAL = "mathematical"             # 数学计算
    FACT_VERIFICATION = "fact_verification"   # 事实核查
    INSTRUCTION_FOLLOWING = "instruction_following"  # 指令跟随
    MULTI_TURN_CONVERSATION = "multi_turn"   # 多轮对话
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
    
    # 新增：各维度能力评分（参考MMLU、HumanEval等基准）
    reasoning_score: float = 0.5        # 推理能力
    coding_score: float = 0.5           # 编程能力
    factual_score: float = 0.5          # 事实准确性
    instruction_score: float = 0.5      # 指令遵循能力
    
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
            task_types=[TaskType.LOGIC_REASONING, TaskType.SUMMARIZATION, 
                       TaskType.CLASSIFICATION, TaskType.FACT_VERIFICATION,
                       TaskType.INSTRUCTION_FOLLOWING],
            speed_score=0.6,
            quality_score=0.95,
            cost_per_1k_tokens=0.001,
            reasoning_score=0.92,  # 推理能力强
            coding_score=0.75,
            factual_score=0.88,
            instruction_score=0.90
        ),
        "minimax-m2.5": ModelInfo(
            name="MiniMax M2.5",
            endpoint="minimaxai/minimax-m2.5",
            task_types=[TaskType.CODE_GENERATION, TaskType.DEFAULT, 
                       TaskType.CREATIVE_WRITING, TaskType.MULTI_TURN_CONVERSATION],
            speed_score=0.4,
            quality_score=0.85,
            cost_per_1k_tokens=0.001,
            reasoning_score=0.78,
            coding_score=0.88,  # 编程能力强
            factual_score=0.75,
            instruction_score=0.82
        ),
        "mixtral-8x7b": ModelInfo(
            name="Mixtral 8x7B",
            endpoint="mistralai/mixtral-8x7b-instruct-v0.1",
            task_types=[TaskType.QUESTION_ANSWER, TaskType.DEFAULT, 
                       TaskType.TRANSLATION, TaskType.MATHEMATICAL],
            speed_score=0.9,   # 速度最快
            quality_score=0.75,
            cost_per_1k_tokens=0.0007,
            reasoning_score=0.70,
            coding_score=0.72,
            factual_score=0.68,
            instruction_score=0.75
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
        
        # 任务分类关键词映射 - 扩展更多场景
        self.task_keywords = {
            TaskType.LOGIC_REASONING: ["逻辑", "推理", "计算", "解方程", "证明", "分析", "推导", "判断"],
            TaskType.CODE_GENERATION: ["代码", "写程序", "函数", "算法", "实现", "编程", "Python", "JavaScript", "写个"],
            TaskType.QUESTION_ANSWER: ["什么是", "为什么", "怎么样", "如何", "问题", "问答", "解释"],
            TaskType.SUMMARIZATION: ["总结", "概括", "摘要", "核心观点", "要点", "简短描述"],
            TaskType.CREATIVE_WRITING: ["写", "创作", "故事", "诗歌", "文章", "文案", "写一首"],
            TaskType.TRANSLATION: ["翻译", "英文", "中文", "语言转换", "译成"],
            TaskType.CLASSIFICATION: ["分类", "判断", "属于", "类型", "归类"],
            TaskType.MATHEMATICAL: ["数学", "计算", "加减乘除", "平方", "开方", "求和", "概率"],
            TaskType.FACT_VERIFICATION: ["验证", "真假", "是否正确", "事实", "核实"],
            TaskType.INSTRUCTION_FOLLOWING: ["按照", "遵循", "按照以下", "请务必", "严格"],
            TaskType.MULTI_TURN: ["接着", "然后呢", "还有", "继续", "上文"],
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
        prefer_quality: bool = True,
        capability_weight: str = "reasoning"  # 新增：能力权重
    ) -> ModelInfo:
        """
        根据任务类型选择最优模型
        
        Args:
            task_type: 任务类型
            prefer_speed: 是否优先考虑速度
            prefer_quality: 是否优先考虑质量
            capability_weight: 能力权重 (reasoning/coding/factual/instruction)
            
        Returns:
            ModelInfo: 选中的模型信息
        """
        # 筛选支持该任务类型的模型
        candidates = [
            m for m in self.MODELS.values() 
            if task_type in m.task_types or TaskType.DEFAULT in m.task_types
        ]
        
        if not candidates:
            candidates = [self.MODELS["minimax-m2.5"]]
        
        # 根据能力权重调整排序
        def capability_key(m: ModelInfo) -> float:
            if capability_weight == "reasoning":
                return m.reasoning_score
            elif capability_weight == "coding":
                return m.coding_score
            elif capability_weight == "factual":
                return m.factual_score
            elif capability_weight == "instruction":
                return m.instruction_score
            else:
                return m.综合评分
        
        # 根据偏好排序
        if prefer_speed:
            candidates.sort(key=lambda m: m.speed_score, reverse=True)
        elif prefer_quality:
            candidates.sort(key=capability_key, reverse=True)
        else:
            candidates.sort(key=lambda m: m.综合评分, reverse=True)
        
        # 考虑历史表现：如果某模型历史评分低于0.5，降低优先级
        selected = candidates[0]
        for model in candidates:
            history = self.performance_history.get(model.name, [])
            if history:
                avg_score = sum(history) / len(history)
                if avg_score < 0.5:
                    continue
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


# ==================== 测试用例（难度提升版）====================

def test_model_router():
    """测试模型路由器 - 难度提升版"""
    print("=" * 60)
    print("测试：ModelRouter - 难度提升版")
    print("=" * 60)
    
    router = ModelRouter()
    
    # ========== 第一部分：基础任务分类测试 ==========
    print("\n【第一部分】基础任务分类")
    print("-" * 40)
    
    basic_tests = [
        ("帮我解这道数学题：有3个红箱子...", TaskType.LOGIC_REASONING),
        ("写一个Python算法...", TaskType.CODE_GENERATION),
        ("今天天气怎么样？", TaskType.QUESTION_ANSWER),
        ("总结一下这篇文档...", TaskType.SUMMARIZATION),
    ]
    
    for user_input, expected in basic_tests:
        actual = router.classify_task(user_input)
        status = "✓" if actual == expected else "✗"
        print(f"{status} {user_input[:25]}... => {actual.value}")
    
    # ========== 第二部分：模糊/歧义测试 ==========
    print("\n【第二部分】模糊/歧义任务")
    print("-" * 40)
    
    ambiguous_tests = [
        # 双重否定
        ("不要不回答这个问题", TaskType.QUESTION_ANSWER),
        # 混合任务
        ("帮我写个程序并解释原理", TaskType.CODE_GENERATION),
        # 隐含任务
        ("这段代码有bug吗？", TaskType.CODE_GENERATION),  # 其实是问bug，属于代码相关
        # 缩写/简写
        ("py写个快排", TaskType.CODE_GENERATION),
    ]
    
    for user_input, expected in ambiguous_tests:
        actual = router.classify_task(user_input)
        status = "✓" if actual == expected else "✗"
        print(f"{status} {user_input[:25]}... => {actual.value}")
    
    # ========== 第三部分：能力匹配测试 ==========
    print("\n【第三部分】能力匹配")
    print("-" * 40)
    
    # 测试不同能力权重下的模型选择
    capability_tests = [
        ("解这个逻辑谜题", "reasoning"),
        ("写一个排序算法", "coding"),
        ("验证这个说法是否正确", "factual"),
        ("严格按照以下要求执行", "instruction"),
    ]
    
    for user_input, capability in capability_tests:
        model = router.select_model(
            router.classify_task(user_input),
            capability_weight=capability
        )
        print(f"  任务: {user_input[:20]}...")
        print(f"    能力权重: {capability} => 选用: {model.name}")
        print(f"    能力分: 推理{model.reasoning_score}, 编程{model.coding_score}, 事实{model.factual_score}")
    
    # ========== 第四部分：边界测试 ==========
    print("\n【第四部分】边界测试")
    print("-" * 40)
    
    edge_cases = [
        ("", TaskType.DEFAULT),  # 空输入
        ("   ", TaskType.DEFAULT),  # 空白
        ("abc123xyz", TaskType.DEFAULT),  # 无意义字符
        ("请问", TaskType.QUESTION_ANSWER),  # 极短但有意义
    ]
    
    for user_input, expected in edge_cases:
        actual = router.classify_task(user_input)
        status = "✓" if actual == expected else "✗"
        print(f"{status} '{user_input[:15] if user_input else '(空)'}' => {actual.value}")
    
    # ========== 第五部分：性能追踪测试 ==========
    print("\n【第五部分】性能追踪")
    print("-" * 40)
    
    # 模拟多次评分记录
    router.record_performance("Llama 3.3 70B", 0.85)
    router.record_performance("Llama 3.3 70B", 0.92)
    router.record_performance("Llama 3.3 70B", 0.45)  # 一次低分
    
    history = router.performance_history.get("Llama 3.3 70B", [])
    avg = sum(history) / len(history) if history else 0
    print(f"  历史评分: {history}")
    print(f"  平均分: {avg:.2f}")
    
    # 验证低分后降低优先级
    model = router.select_model(TaskType.LOGIC_REASONING)
    print(f"  低分后选用模型: {model.name}")
    
    # ========== 第六部分：统计信息 ==========
    print("\n【第六部分】统计信息")
    print("-" * 40)
    
    stats = router.get_stats()
    print(f"  模型数量: {stats['total_models']}")
    print(f"  支持任务类型: {len(stats['supported_task_types'])}种")
    print(f"  历史记录: {stats['performance_records']}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_model_router()