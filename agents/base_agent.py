from openai import OpenAI
from config import config
import json
import re

class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        
        # Новый клиент для OpenRouter
        self.client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL
        )
        
        self.system_prompt = system_prompt
    
    def call_llm(self, prompt: str, temperature: float = None) -> str:
        """Вызов LLM через OpenRouter"""
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка при обращении к OpenRouter API: {e}"
    
    def extract_json(self, text: str) -> dict:
        """Извлекает JSON из текста ответа (без изменений)"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"raw_response": text}
        except:
            return {"raw_response": text}