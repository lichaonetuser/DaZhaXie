"""自动评分系统"""
from dataclasses import dataclass
from typing import List

@dataclass
class ScoreResult:
    total_score: float
    self_consistency: float
    fact_accuracy: float
    llm_judge_score: float
    anomaly_score: float
    style_match: float

class AutoScorer:
    def __init__(self, user_style=None):
        self.user_style = user_style or {"avg_length": 500}
    
    async def score(self, question: str, response: str, model=None) -> ScoreResult:
        self_consistency = 0.85
        fact_accuracy = await self._check_fact_accuracy(response)
        llm_score = await self._llm_judge(question, response) if model else 0.7
        anomaly = self._check_anomaly(response)
        style = self._check_style_match(response)
        
        total = self_consistency*0.15 + fact_accuracy*0.25 + llm_score*0.30 + anomaly*0.15 + style*0.15
        
        return ScoreResult(
            total_score=round(total, 3),
            self_consistency=round(self_consistency, 3),
            fact_accuracy=round(fact_accuracy, 3),
            llm_judge_score=round(llm_score, 3),
            anomaly_score=round(anomaly, 3),
            style_match=round(style, 3)
        )
    
    async def _check_fact_accuracy(self, response: str) -> float:
        return 0.85
    
    async def _llm_judge(self, question: str, response: str) -> float:
        return 0.8
    
    def _check_anomaly(self, response: str) -> float:
        if len(response) < 10: return 0.3
        return 0.95
    
    def _check_style_match(self, response: str) -> float:
        return 0.85
