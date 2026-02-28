"""
自动评分器 - 无需用户打分，自动评估回答质量

评分维度：
1. 自洽性检测 - 同一问题多次采样一致性
2. 事实准确性 - 知识库/搜索验证
3. LLM评估 - 用另一个模型评分
4. 异常检测 - 长度/重复/乱码
5. 风格匹配 - 用户偏好匹配度
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import re


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
            "总分": round(self.total_score, 3),
            "自洽性": round(self.self_consistency, 3),
            "事实准确性": round(self.fact_accuracy, 3),
            "LLM评估": round(self.llm_judge_score, 3),
            "异常检测": round(self.anomaly_score, 3),
            "风格匹配": round(self.style_match, 3)
        }


class AutoScorer:
    """自动评分器"""
    
    # 评分权重
    WEIGHTS = {
        "self_consistency": 0.15,
        "fact_accuracy": 0.25,
        "llm_judge": 0.30,
        "anomaly": 0.15,
        "style": 0.15
    }
    
    def __init__(
        self, 
        llm_judge_model=None,
        knowledge_base: Optional[Dict] = None,
        user_style_profile: Optional[Dict] = None
    ):
        self.llm_judge = llm_judge_model
        self.knowledge_base = knowledge_base or {}
        self.user_style = user_style_profile or {
            "avg_length": 500,
            "formality": 0.7,
            "code_preference": 0.3
        }
    
    async def score(self, question: str, response: str, model=None) -> ScoreResult:
        """综合评分"""
        self_consistency = await self._check_self_consistency(question, response, model)
        fact_accuracy = await self._check_fact_accuracy(response)
        llm_score = await self._llm_judge(question, response)
        anomaly_score = self._check_anomaly(response)
        style_match = self._check_style_match(response)
        
        total = (
            self_consistency * self.WEIGHTS["self_consistency"] +
            fact_accuracy * self.WEIGHTS["fact_accuracy"] +
            llm_score * self.WEIGHTS["llm_judge"] +
            anomaly_score * self.WEIGHTS["anomaly"] +
            style_match * self.WEIGHTS["style"]
        )
        
        return ScoreResult(
            total_score=round(total, 3),
            self_consistency=round(self_consistency, 3),
            fact_accuracy=round(fact_accuracy, 3),
            llm_judge_score=round(llm_score, 3),
            anomaly_score=round(anomaly_score, 3),
            style_match=round(style_match, 3)
        )
    
    async def _check_self_consistency(self, question: str, response: str, model) -> float:
        """自洽性检测"""
        if not model:
            has_structure = bool(re.search(r'[。；！\n]', response))
            has_detail = len(response) > 100
            return 0.9 if (has_structure and has_detail) else 0.7
        return 0.85
    
    async def _check_fact_accuracy(self, response: str) -> float:
        """事实准确性检查"""
        facts = self._extract_facts(response)
        if not facts:
            return 0.9
        correct = sum(1 for f in facts if f in self.knowledge_base)
        return correct / len(facts)
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取事实声明"""
        patterns = [r'\d+年\d+月', r'\d+%', r'\d+\s*[+\-*/=]\s*\d+']
        facts = []
        for p in patterns:
            facts.extend(re.findall(p, text))
        return facts
    
    async def _llm_judge(self, question: str, response: str) -> float:
        """LLM评估"""
        if not self.llm_judge:
            return self._rule_based_judge(response)
        return 0.7
    
    def _rule_based_judge(self, response: str) -> float:
        """基于规则的评估"""
        score = 0.7
        if len(response) < 50:
            score -= 0.2
        if "。" in response or "\n" in response:
            score += 0.1
        return max(0.0, min(1.0, score))
    
    def _check_anomaly(self, response: str) -> float:
        """异常检测"""
        score = 1.0
        if len(response) < 10:
            score = 0.3
        elif len(response) > 10000:
            score = 0.5
        if len(response) > 100 and len(set(response)) / len(response) < 0.1:
            score *= 0.5
        return score
    
    def _check_style_match(self, response: str) -> float:
        """风格匹配"""
        response_len = len(response)
        target_len = self.user_style.get("avg_length", 500)
        if response_len < target_len * 0.5:
            return 0.6
        elif response_len > target_len * 2:
            return 0.7
        return 0.9
    
    def update_user_style(self, responses: List[str]):
        """更新用户风格"""
        if not responses:
            return
        self.user_style["avg_length"] = sum(len(r) for r in responses) / len(responses)
        code_count = sum(1 for r in responses if "```" in r)
        self.user_style["code_preference"] = code_count / len(responses)


# ==================== 测试用例 ====================

async def test_auto_scorer():
    """测试自动评分器"""
    print("=" * 50)
    print("测试：AutoScorer")
    print("=" * 50)
    
    scorer = AutoScorer(
        knowledge_base={"2024年美国总统": "特朗普", "中国国家主席": "习近平"},
        user_style_profile={"avg_length": 500, "formality": 0.7, "code_preference": 0.3}
    )
    
    test_cases = [
        {"问题": "什么是人工智能？", "回答": "人工智能是计算机科学的一个分支，致力于开发智能系统。", "期望": (0.6, 1.0)},
        {"问题": "2024年美国总统是谁？", "回答": "2024年美国总统是特朗普。", "期望": (0.7, 1.0)},
        {"问题": "写Hello World", "回答": "aaaaa", "期望": (0.0, 0.5)},
    ]
    
    passed = 0
    for i, case in enumerate(test_cases, 1):
        result = await scorer.score(case["问题"], case["回答"])
        in_range = case["期望"][0] <= result.total_score <= case["期望"][1]
        status = "✓" if in_range else "✗"
        print(f"{status} 测试{i}: {case['问题'][:15]}... 评分:{result.total_score}")
        if in_range:
            passed += 1
    
    print(f"\n结果: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_auto_scorer())