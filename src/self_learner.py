"""自我学习系统"""
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime
import json

@dataclass
class CaseRecord:
    id: str
    question: str
    response: str
    score: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class SelfLearner:
    def __init__(self, storage_path="/home/codespace/.openclaw/learner"):
        self.storage_path = storage_path
        self.cases: Dict[str, CaseRecord] = {}
        self.errors: Dict[str, dict] = {}
    
    def learn(self, question: str, response: str, score_result):
        case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if score_result.total_score >= 0.8:
            self.cases[case_id] = CaseRecord(id=case_id, question=question, response=response, score=score_result.total_score)
            print(f"✓ 成功案例入库: {case_id}")
        elif score_result.total_score <= 0.4:
            self.errors[case_id] = {"question": question, "response": response, "score": score_result.total_score}
            print(f"✗ 错误记录入库: {case_id}")
    
    def get_stats(self) -> Dict:
        return {"total_cases": len(self.cases), "total_errors": len(self.errors)}
    
    def get_similar_cases(self, question: str, limit: int = 5) -> List[CaseRecord]:
        keywords = set(question.lower().split())
        scored = []
        for case in self.cases.values():
            overlap = len(keywords & set(case.question.lower().split()))
            if overlap > 0:
                scored.append((overlap, case))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]
