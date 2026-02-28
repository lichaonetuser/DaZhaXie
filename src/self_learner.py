"""
自我学习器 - 根据评分自动优化

功能：
1. 成功案例学习 - 高分回答存入案例库
2. 错误记录学习 - 低分回答存入错误库
3. 模式提取 - 从案例中提取模式标签
4. Prompt改进建议 - 连续低分触发改进
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CaseRecord:
    """成功案例记录"""
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
    correct_response: str = ""
    error_type: str = ""  # fact_error, logic_error, understanding_error
    root_cause: str = ""
    lesson: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PromptImprovement:
    """Prompt改进建议"""
    id: str
    current_prompt: str = ""
    suggested_change: str = ""
    reason: str = ""
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SelfLearner:
    """自我学习器"""
    
    def __init__(self, storage_path: str = "~/.openclaw/learner"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.cases: Dict[str, CaseRecord] = self._load_cases()
        self.errors: Dict[str, ErrorRecord] = self._load_errors()
        self.prompt_improvements: Dict[str, PromptImprovement] = self._load_improvements()
    
    def _load_cases(self) -> Dict[str, CaseRecord]:
        path = self.storage_path / "cases.json"
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
            return {k: CaseRecord(**v) for k, v in data.items()}
    
    def _save_cases(self):
        path = self.storage_path / "cases.json"
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self.cases.items()}, f, ensure_ascii=False, indent=2)
    
    def _load_errors(self) -> Dict[str, ErrorRecord]:
        path = self.storage_path / "errors.json"
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
            return {k: ErrorRecord(**v) for k, v in data.items()}
    
    def _save_errors(self):
        path = self.storage_path / "errors.json"
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self.errors.items()}, f, ensure_ascii=False, indent=2)
    
    def _load_improvements(self) -> Dict[str, PromptImprovement]:
        path = self.storage_path / "prompt_improvements.json"
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
            return {k: PromptImprovement(**v) for k, v in data.items()}
    
    def _save_improvements(self):
        path = self.storage_path / "prompt_improvements.json"
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self.prompt_improvements.items()}, f, ensure_ascii=False, indent=2)
    
    def learn(self, question: str, response: str, score_result, correct_response: Optional[str] = None):
        """学习入口"""
        case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if score_result.total_score >= 0.8:
            self._learn_success(case_id, question, response, score_result)
        elif score_result.total_score <= 0.4:
            self._learn_failure(case_id, question, response, correct_response, score_result)
    
    def _learn_success(self, case_id: str, question: str, response: str, score_result):
        """从成功案例学习"""
        indicators = []
        if score_result.self_consistency > 0.8:
            indicators.append("自洽")
        if score_result.fact_accuracy > 0.9:
            indicators.append("准确")
        if score_result.llm_judge_score > 0.8:
            indicators.append("高质量")
        
        tags = self._extract_pattern_tags(question, response)
        
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
        print(f"✓ 成功案例入库: {case_id}, 得分:{score_result.total_score}")
    
    def _learn_failure(self, case_id: str, question: str, response: str, correct_response: Optional[str], score_result):
        """从失败案例学习"""
        error_type = self._classify_error(score_result)
        lesson = self._extract_lesson(score_result)
        
        error = ErrorRecord(
            id=case_id,
            question=question,
            wrong_response=response,
            correct_response=correct_response or "",
            error_type=error_type,
            root_cause=self._analyze_root_cause(score_result),
            lesson=lesson
        )
        
        self.errors[case_id] = error
        self._save_errors()
        self._check_prompt_improvement(case_id, question, score_result)
        print(f"✗ 错误记录入库: {case_id}, 类型:{error_type}")
    
    def _extract_pattern_tags(self, question: str, response: str) -> List[str]:
        """提取模式标签"""
        tags = []
        q = question.lower()
        if any(w in q for w in ["怎么", "如何", "步骤"]):
            tags.append("操作指南")
        elif any(w in q for w in ["是什么", "定义"]):
            tags.append("概念解释")
        elif any(w in q for w in ["为什么", "原因"]):
            tags.append("原因分析")
        
        if "```" in response:
            tags.append("含代码")
        if len(response) > 1000:
            tags.append("详细")
        if "首先" in response and "然后" in response:
            tags.append("结构化")
        
        return tags
    
    def _classify_error(self, score_result) -> str:
        """错误分类"""
        if score_result.fact_accuracy < 0.5:
            return "事实错误"
        elif score_result.self_consistency < 0.5:
            return "逻辑错误"
        elif score_result.anomaly_score < 0.5:
            return "输出异常"
        return "理解错误"
    
    def _extract_lesson(self, score_result) -> str:
        """提取教训"""
        lessons = []
        if score_result.fact_accuracy < 0.5:
            lessons.append("需加强事实验证")
        if score_result.self_consistency < 0.5:
            lessons.append("需检查推理自洽性")
        if score_result.style_match < 0.5:
            lessons.append("需匹配用户风格")
        return "; ".join(lessons) if lessons else "需进一步分析"
    
    def _analyze_root_cause(self, score_result) -> str:
        """分析根因"""
        if score_result.llm_judge_score < score_result.total_score:
            return "整体质量不足"
        return "特定维度存在问题"
    
    def _check_prompt_improvement(self, case_id: str, question: str, score_result):
        """检查是否需要Prompt改进"""
        recent_poor = [c for c in self.cases.values() if c.score <= 0.4][-3:]
        
        if len(recent_poor) >= 3:
            imp_id = f"imp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            improvement = PromptImprovement(
                id=imp_id,
                suggested_change="建议增强特定场景的提示词",
                reason="连续3次低分，需检查Prompt清晰度"
            )
            self.prompt_improvements[imp_id] = improvement
            self._save_improvements()
            print(f"⚠️ 建议优化Prompt: {imp_id}")
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            "总案例数": len(self.cases),
            "总错误数": len(self.errors),
            "优秀案例": sum(1 for c in self.cases.values() if c.score >= 0.8),
            "错误案例": sum(1 for c in self.cases.values() if c.score <= 0.4)
        }
    
    def get_similar_cases(self, question: str, limit: int = 5) -> List[CaseRecord]:
        """获取相似案例"""
        keywords = set(question.lower().split())
        scored = []
        for case in self.cases.values():
            case_kw = set(case.question.lower().split())
            overlap = len(keywords & case_kw)
            if overlap > 0:
                scored.append((overlap, case))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]
    
    def get_error_patterns(self) -> Dict[str, int]:
        """获取错误模式统计"""
        patterns = {}
        for e in self.errors.values():
            patterns[e.error_type] = patterns.get(e.error_type, 0) + 1
        return patterns


# ==================== 测试用例 ====================

def test_self_learner():
    """测试自我学习器"""
    print("=" * 50)
    print("测试：SelfLearner")
    print("=" * 50)
    
    learner = SelfLearner()
    
    # 模拟高分案例
    from dataclasses import SimpleNamespace
    good_score = SimpleNamespace(
        total_score=0.85,
        self_consistency=0.9,
        fact_accuracy=0.95,
        llm_judge_score=0.88,
        anomaly_score=0.95,
        style_match=0.8
    )
    
    # 模拟低分案例
    bad_score = SimpleNamespace(
        total_score=0.35,
        self_consistency=0.4,
        fact_accuracy=0.3,
        llm_judge_score=0.4,
        anomaly_score=0.9,
        style_match=0.5
    )
    
    # 学习成功案例
    print("\n--- 学习成功案例 ---")
    learner.learn("如何用Python实现快排？", "def quick_sort(arr):...", good_score)
    
    # 学习错误案例
    print("\n--- 学习错误案例 ---")
    learner.learn("2024年美国总统是谁？", "是拜登", bad_score, correct_response="是特朗普")
    
    # 查看统计
    print("\n--- 统计信息 ---")
    stats = learner.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # 相似案例查询
    print("\n--- 相似案例查询 ---")
    similar = learner.get_similar_cases("Python排序算法")
    print(f"找到 {len(similar)} 个相似案例")
    
    # 错误模式分析
    print("\n--- 错误模式分析 ---")
    patterns = learner.get_error_patterns()
    for t, c in patterns.items():
        print(f"  {t}: {c}次")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    test_self_learner()