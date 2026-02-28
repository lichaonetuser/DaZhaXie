#!/usr/bin/env python3
"""
NVIDIA NIM 模型评测脚本
测试各模型的对话能力、响应速度、幻觉率、复杂问题处理能力
"""

import time
import json
import os

# 测试用例
TEST_CASES = {
    "logic": {
        "name": "逻辑推理",
        "prompt": """有3个红箱子和5个蓝箱子，每个蓝箱子比红箱子多2个球。已知总球数27，求红蓝箱子各有多少球？请给出完整推理过程。"""
    },
    "multi_hop": {
        "name": "多跳关联推理", 
        "prompt": """小明是小李的老师，小李是小王的父亲。小王的爷爷是谁？请列出所有可能答案，并解释原因。"""
    },
    "hallucination": {
        "name": "幻觉测试（事实核查）",
        "prompt": """请列出以下人物的详细信息：1. 2024年美国总统 2. 中国现任国家主席 3. 特斯拉CEO 包含姓名、政党/职务、任期时间"""
    },
    "coding": {
        "name": "代码算法",
        "prompt": """写一个Python算法，找出数组中出现次数超过⌊n/3⌋的所有元素，要求时间复杂度O(n)，空间复杂度O(1)。包含完整代码和测试用例。"""
    },
    "ambiguous": {
        "name": "模糊指令理解",
        "prompt": """'意思意思'这句话在以下不同语境下的含义：1. 商务场合 2. 朋友聊天 3. 家人之间 4. 职场请客 分别解释其含义。"""
    },
    "long_context": {
        "name": "长上下文理解",
        "prompt": """根据以下文章摘要总结核心观点：

摘要1：本文探讨了人工智能在医疗诊断中的应用，指出AI在影像识别领域已超越人类医生水平，但在复杂病例判断上仍需人类专家监督。

摘要2：研究表明，AI辅助诊断可以减少40%的误诊率，但患者对AI的信任度仍较低，多数人更倾向于人类医生做出最终诊断。

摘要3：医疗AI面临的主要挑战包括数据隐私、算法透明度以及伦理责任归属问题，这些都需要制定明确的监管框架。

请总结上述三篇摘要的共同核心观点。"""
    }
}

# NVIDIA NIM API 配置
# 需要设置环境变量 NVIDIA_API_KEY
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = {
    "minimax-m2.5": {
        "name": "MiniMax M2.5",
        "endpoint": f"{NIM_BASE_URL}/chat/completions",
        "model_id": "minimaxai/minimax-m2.5"
    },
    "llama-3.3-70b": {
        "name": "Llama 3.3 70B", 
        "endpoint": f"{NIM_BASE_URL}/chat/completions",
        "model_id": "meta/llama-3.3-70b-instruct"
    },
    "mistral-large": {
        "name": "Mistral Large",
        "endpoint": f"{NIM_BASE_URL}/chat/completions", 
        "model_id": "mistralai/mistral-large-2411"
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B",
        "endpoint": f"{NIM_BASE_URL}/chat/completions",
        "model_id": "mistralai/mixtral-8x7b-instruct-v0.1"
    },
    "qwen2.5-72b": {
        "name": "Qwen 2.5 72B",
        "endpoint": f"{NIM_BASE_URL}/chat/completions",
        "model_id": "qwen/qwen2.5-72b-instruct"
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "endpoint": f"{NIM_BASE_URL}/chat/completions",
        "model_id": "deepseek-ai/deepseek-r1"
    },
    "phi-4-mini": {
        "name": "Phi-4 Mini",
        "endpoint": f"{NIM_BASE_URL}/chat/completions",
        "model_id": "microsoft/phi-4-mini-instruct"
    }
}


def call_nim_api(model_id: str, messages: list, timeout: int = 120) -> dict:
    """调用 NVIDIA NIM API"""
    import requests
    
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return {"error": "NVIDIA_API_KEY not set"}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "elapsed": elapsed,
                "usage": result.get("usage", {})
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "elapsed": elapsed
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start_time
        }


def evaluate_response(test_type: str, response: str) -> dict:
    """评估回答质量"""
    evaluation = {
        "logic": {
            "score": 0,
            "reasoning_steps": 0,
            "correctness": "unknown",
            "issues": []
        },
        "multi_hop": {
            "score": 0,
            "possibilities_identified": 0,
            "reasoning": "unknown"
        },
        "hallucination": {
            "score": 100,
            "factual_errors": [],
            "accuracy": "unknown"
        },
        "coding": {
            "score": 0,
            "has_code": False,
            "has_tests": False,
            "correct_complexity": False
        },
        "ambiguous": {
            "score": 0,
            "contexts_covered": 0
        },
        "long_context": {
            "score": 0,
            "points_summarized": 0,
            "coherence": "unknown"
        }
    }
    
    # 简单规则评估（实际应该用更复杂的逻辑）
    eval_cfg = evaluation[test_type]
    
    if test_type == "logic":
        # 检查是否有推理步骤
        if "解：" in response or "设" in response or "方程" in response:
            eval_cfg["reasoning_steps"] = response.count("\n")
            eval_cfg["score"] = min(100, eval_cfg["reasoning_steps"] * 10)
        if "红箱子" in response and "蓝箱子" in response:
            eval_cfg["correctness"] = "likely_correct"
            
    elif test_type == "multi_hop":
        if "爷爷" in response:
            eval_cfg["possibilities_identified"] = response.count("可能")
            eval_cfg["score"] = min(100, eval_cfg["possibilities_identified"] * 30)
            
    elif test_type == "hallucination":
        # 检查幻觉（2024年美国总统应该是拜登，但2024年大选结果未知）
        factual_errors = []
        if "2024年美国总统" in response:
            if "未知" not in response and "不确定" not in response:
                # 如果不确定但给出确切答案，可能是幻觉
                factual_errors.append("对2024年美国总统给出确定性答案")
        eval_cfg["factual_errors"] = factual_errors
        eval_cfg["score"] = max(0, 100 - len(factual_errors) * 30)
        
    elif test_type == "coding":
        if "def " in response or "class " in response:
            eval_cfg["has_code"] = True
            eval_cfg["score"] = 50
        if "test" in response.lower() or "assert" in response:
            eval_cfg["has_tests"] = True
            eval_cfg["score"] += 20
        if "O(n)" in response:
            eval_cfg["correct_complexity"] = True
            eval_cfg["score"] += 30
            
    elif test_type == "ambiguous":
        contexts = ["商务", "朋友", "家人", "职场"]
        covered = sum(1 for c in contexts if c in response)
        eval_cfg["contexts_covered"] = covered
        eval_cfg["score"] = covered * 25
        
    elif test_type == "long_context":
        if len(response) > 50:
            eval_cfg["points_summarized"] = min(5, len(response) // 50)
            eval_cfg["score"] = eval_cfg["points_summarized"] * 20
            
    return eval_cfg


def run_test(model_id: str, test_type: str, test_case: dict) -> dict:
    """运行单个测试"""
    print(f"\n测试 {test_case['name']}...")
    
    messages = [{"role": "user", "content": test_case["prompt"]}]
    
    result = call_nim_api(model_id, messages)
    
    if not result.get("success"):
        return {
            "test": test_type,
            "name": test_case["name"],
            "success": False,
            "error": result.get("error"),
            "elapsed": result.get("elapsed", 0)
        }
    
    evaluation = evaluate_response(test_type, result["content"])
    
    return {
        "test": test_type,
        "name": test_case["name"],
        "success": True,
        "response": result["content"][:500],  # 截断保存
        "elapsed": result["elapsed"],
        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
        "evaluation": evaluation
    }


def generate_report(model_name: str, results: list) -> str:
    """生成评测报告"""
    report = []
    report.append(f"# {model_name} 评测报告")
    report.append("")
    report.append("## 总览")
    
    total_time = sum(r.get("elapsed", 0) for r in results if r.get("success"))
    total_tokens = sum(r.get("tokens_used", 0) for r in results if r.get("success"))
    success_count = sum(1 for r in results if r.get("success"))
    
    report.append(f"- 测试用例数: {len(results)}")
    report.append(f"- 成功数: {success_count}")
    report.append(f"- 总耗时: {total_time:.2f}秒")
    report.append(f"- 总Token消耗: {total_tokens}")
    report.append("")
    report.append("## 详细结果")
    
    for r in results:
        report.append(f"### {r['name']}")
        report.append(f"- 耗时: {r.get('elapsed', 0):.2f}秒")
        report.append(f"- 状态: {'✅ 成功' if r.get('success') else '❌ 失败'}")
        
        if r.get("success"):
            eval_data = r.get("evaluation", {})
            report.append(f"- 评分: {eval_data.get('score', 0)}/100")
            
            if r["test"] == "hallucination":
                errors = eval_data.get("factual_errors", [])
                report.append(f"- 事实错误: {len(errors)}个")
                if errors:
                    for e in errors:
                        report.append(f"  - {e}")
                        
            elif r["test"] == "coding":
                report.append(f"- 含代码: {'是' if eval_data.get('has_code') else '否'}")
                report.append(f"- 含测试: {'是' if eval_data.get('has_tests') else '否'}")
                report.append(f"- 复杂度正确: {'是' if eval_data.get('correct_complexity') else '否'}")
                
        else:
            report.append(f"- 错误: {r.get('error', 'Unknown')}")
            
        report.append("")
    
    return "\n".join(report)


def main():
    """主函数"""
    # 检查 API Key
    if not os.environ.get("NVIDIA_API_KEY"):
        print("请设置 NVIDIA_API_KEY 环境变量")
        print("export NVIDIA_API_KEY=your_api_key")
        return
    
    # 选择要测试的模型（从MODELS字典选择）
    models_to_test = ["minimax-m2.5"]  # 可以添加更多
    
    all_results = {}
    
    for model_key in models_to_test:
        model_info = MODELS.get(model_key)
        if not model_info:
            print(f"未知模型: {model_key}")
            continue
            
        print(f"\n{'='*50}")
        print(f"测试模型: {model_info['name']}")
        print(f"{'='*50}")
        
        results = []
        for test_type, test_case in TEST_CASES.items():
            result = run_test(model_info["model_id"], test_type, test_case)
            results.append(result)
            
        all_results[model_key] = results
        
        # 生成报告
        report = generate_report(model_info["name"], results)
        print(report)
        
        # 保存报告
        report_path = f"/home/codespace/.openclaw/workspace/report_{model_key}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()