import os
import ast
import re

from llm_client import HelloAgentsLLM
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量，处理文件不存在异常
try:
    load_dotenv()
except FileNotFoundError:
    print("警告：未找到 .env 文件，将使用系统环境变量。")
except Exception as e:
    print(f"警告：加载 .env 文件时出错: {e}")

# --- 1. LLM客户端定义 ---
# 假设你已经有llm_client.py文件，里面定义了HelloAgentsLLM类

# --- 2. 规划器 (Planner) 定义 ---
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")
        
        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()

            # jlq_fix ==========================================================
            # 2. 【关键步骤】自动修复缺失的引号
            # 逻辑：查找 [ 或 , 后面，没有以 " 或 ' 开头，但以汉字/字母开头的片段，强制加上双引号
            # 正则解释：
            # ([,\[])      -> 捕获组1：逗号 或 左方括号
            # \s*          -> 可选的空白字符
            # (?!["\'])    -> 负向先行断言：确保后面不是引号
            # ([\u4e00-\u9fa5a-zA-Z][^,\]]*?) -> 捕获组2：以汉字或字母开头，直到遇到逗号或右括号前的内容
            # \s*([,\]])   -> 捕获组3：可选空白 + 逗号 或 右方括号

            pattern = r'([,\[])\s*(?!["\'])([\u4e00-\u9fa5a-zA-Z][^,\]]*?)\s*([,\]])'

            # 循环替换，直到没有匹配项为止（防止嵌套或连续多个错误）
            while re.search(pattern, plan_str):
                plan_str = re.sub(pattern, r'\1 "\2" \3', plan_str)
                print(f"🔧 检测到缺失引号，已修复。当前字符串: {plan_str}")

            # 3. 【步骤 B】关键新增：清洗重复的引号和非法格式
            # 3.1 将连续的两个或多个双引号 "" 替换为一个 "
            # 例如： "...""]  -> "...""]
            plan_str = re.sub(r'"{2,}', '"', plan_str)

            # 3.2 清理引号和括号之间可能产生的多余空格 (可选，但能增加稳定性)
            # 例如： " " ]  -> " "]
            plan_str = re.sub(r'"\s+"', '"', plan_str)
            plan_str = re.sub(r' $  \s* $  ', ']', plan_str)  # 防止出现 ]]

            print(f"✨ 清洗重复符号后: {plan_str}")
            # jlq_fix end=======================================================

            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

# --- 3. 执行器 (Executor) 定义 ---
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        final_answer = ""
        
        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan, history=history if history else "无", current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            
            response_text = self.llm_client.think(messages=messages) or ""
            
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")
            
        return final_answer

# --- 4. 智能体 (Agent) 整合 ---
class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

# --- 5. 主函数入口 ---
if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
        agent = PlanAndSolveAgent(llm_client)
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent.run(question)
    except ValueError as e:
        print(e)
