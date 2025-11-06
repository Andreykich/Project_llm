from .base_agent import BaseAgent
from config import config
from database.json_db import JSONDatabase
import json

class DataAnalyzerAgent(BaseAgent):
    def __init__(self, db: JSONDatabase):
        super().__init__(
            name="Data Analyzer", 
            system_prompt=config.PROMPTS["data_analyzer"]
        )
        self.db = db
    
    def generate_advice(self, user_data: dict) -> str:
        """Генерирует советы на основе данных пользователя и БД"""
        
        # Ищем релевантные тренды в БД
        trends = self.db.search_trends(
            industry=user_data.get('industry', ''),
            city=user_data.get('city', '')
        )
        
        prompt = f"""
        ДАННЫЕ ПОЛЬЗОВАТЕЛЯ:
        {json.dumps(user_data, ensure_ascii=False, indent=2)}
        
        РЕЛЕВАНТНЫЕ ТРЕНДЫ ИЗ БАЗЫ ДАННЫХ:
        {json.dumps(trends, ensure_ascii=False, indent=2)}
        
        Сгенерируй подробные практические советы используя Chain of Thought.
        Структура ответа:
        
        # Анализ ситуации
        [Твой анализ на основе данных]
        
        # Ключевые тренды  
        [Актуальные тренды в этой сфере]
        
        # Пошаговый план
        [Конкретные шаги для запуска]
        
        # Риски и решения
        [Потенциальные проблемы и как их избежать]
        
        # Бюджет и ресурсы
        [Рекомендации по бюджету]
        
        # Маркетинг
        [Стратегия продвижения]
        """
        
        return self.call_llm(prompt, temperature=0.5)