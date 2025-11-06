import requests
from bs4 import BeautifulSoup
from .base_agent import BaseAgent
from config import config
import json

class WebParserAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Web Parser",
            system_prompt=config.PROMPTS["web_parser"]
        )
    
    def parse_website(self, url: str) -> dict:
        """Парсит сайт и извлекает структурированную информацию"""
        
        try:
            # Загружаем страницу
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Извлекаем основной текст
            text = soup.get_text()[:8000]  # Ограничиваем длину
            
            prompt = f"""
            Содержимое веб-страницы с {url}:
            
            {text}
            
            Проанализируй и извлеки информацию о бизнес-трендах, возможностях для стартапов, 
            перспективных нишах. Верни JSON со структурой:
            {{
                "trends": ["тренд1", "тренд2"],
                "opportunities": ["возможность1", "возможность2"], 
                "statistics": {{"статистика1": "значение"}},
                "advice": ["совет1", "совет2"],
                "sources": ["источник1"],
                "industry": "IT/Общепит/Retail/Услуги",
                "confidence": 0.8
            }}
            """
            
            result = self.call_llm(prompt, temperature=0.2)
            return self.extract_json(result)
            
        except Exception as e:
            return {"error": f"Ошибка парсинга: {e}"}