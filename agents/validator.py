from .base_agent import BaseAgent
from config import config
import json

class ValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Validator",
            system_prompt=config.PROMPTS["validator"]
        )
    
    def validate_data(self, data: dict, data_type: str = "trend") -> dict:
        """Валидирует данные перед сохранением в БД"""
        
        prompt = f"""
        Проверь следующие данные (тип: {data_type}):
        {json.dumps(data, ensure_ascii=False, indent=2)}
        
        Критерии валидации:
        1. Актуальность 
        2. Релевантность бизнес-тематике
        3. Конкретность (есть цифры, факты, а не общие слова)
        4. Практическая применимость
        
        !!!Пока проект находится на стадии тестирования будь подобрее и пусть не проходят проверку только крайне неподхдящая информация

        Верни JSON:
        {{
            "is_valid": true/false,
            "issues": ["проблема1", "проблема2"],
            "confidence_score": 0.95,
            "suggestions": ["предложение1"],
            "validated_fields": ["поле1", "поле2"]
        }}
        """
        
        result = self.call_llm(prompt, temperature=0.1)
        return self.extract_json(result)
    

    # Критерии валидации:
    #     1. Актуальность (данные не старше 2 лет)
    #     2. Достоверность (есть ссылки на источники) 
    #     3. Релевантность бизнес-тематике
    #     4. Конкретность (есть цифры, факты, а не общие слова)
    #     5. Практическая применимость