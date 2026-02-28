"""
Agent 智能优化系统 - 主入口

整合三个核心模块：
1. ModelRouter - 模型路由器
2. AutoScorer - 自动评分器  
3. SelfLearner - 自我学习器
"""

from src.model_router import ModelRouter
from src.auto_scorer import AutoScorer
from src.self_learner import SelfLearner


class AgentOptimizer:
    """Agent智能优化系统"""
    
    def __init__(self):
        self.router = ModelRouter()
        self.scorer = AutoScorer()
        self.learner = SelfLearner()
    
    async def process(self, user_input: str) -> dict:
        model = self.router.route(user_input)
        response = f"[使用模型: {model.name}] 这是一个模拟回答。"
        score_result = await self.scorer.score(user_input, response)
        self.learner.learn(user_input, response, score_result)
        self.router.record_performance(model.name, score_result.total_score)
        return {
            "response": response,
            "model_used": model.name,
            "score": score_result.to_dict()
        }


def test_full_system():
    print("=" * 60)
    print("测试：Agent 智能优化系统")
    print("=" * 60)
    
    optimizer = AgentOptimizer()
    test_cases = [
        "帮我解这道数学题",
        "写一个Python快排",
        "今天天气怎么样？",
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i}: {user_input} ---")
        model = optimizer.router.route(user_input)
        print(f"选用模型: {model.name}")
    
    print("\n" + "=" * 60)
    print(f"路由统计: {optimizer.router.get_stats()}")
    print(f"学习统计: {optimizer.learner.get_stats()}")
    print("测试完成!")


if __name__ == "__main__":
    test_full_system()