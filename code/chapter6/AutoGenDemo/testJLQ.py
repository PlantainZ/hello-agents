import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

model = os.getenv("LLM_MODEL_ID"),
api_key = os.getenv("LLM_API_KEY"),
base_url = os.getenv("LLM_BASE_URL")
print(f"This is model:{model}")

# .env模型访问正常