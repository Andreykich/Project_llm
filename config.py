import os
from dotenv import load_dotenv
load_dotenv()  

from dataclasses import dataclass

@dataclass
class Config:
    # OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    # MODEL: str = "gpt-3.5-turbo"  # или "gpt-4"
    # TEMPERATURE: float = 0.7
    # MAX_TOKENS: int = 2000
    
    # # Пути к данным
    # DB_PATH: str = "data/database.json"
    # PARSED_DATA_PATH: str = "data/parsed_data/"
    
  
    # Новое для OpenRouter + DeepSeek
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    MODEL: str = "deepseek/deepseek-v3.2-exp"  # Модель через OpenRouter
    
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Пути к данным
    DB_PATH: str = "data/database.json"
    PARSED_DATA_PATH: str = "data/parsed_data/"


    # Промпты для агентов
    PROMPTS = {
        "data_collector": """Ты - бизнес-консультант. Твоя задача - собрать информацию о бизнес-идее пользователя.
Задавай вопросы по одному, будь дружелюбным. Собери следующую информацию:
1. Сфера деятельности (IT, общепит, retail, услуги и т.д.)
2. Конкретная идея (что именно хочет открыть)
3. Город/регион
4. Бюджет (если известен)
5. Опыт в этой сфере
6. Целевая аудитория
7. Особые пожелания или ограничения

После сбора всей информации, верни JSON с полями: industry, idea, city, budget, experience, target_audience, special_requirements""",
        
        "data_analyzer": """Ты - бизнес-аналитик. На основе данных из базы и информации от пользователя, составь практические советы для стартапа.
Используй Chain of Thought:
1. Сначала проанализируй тренды в этой сфере
2. Выдели ключевые возможности и риски  
3. Предложи конкретные шаги для запуска
4. Учти региональные особенности
5. Дай рекомендации по бюджету и маркетингу
6. Предложи потенциальных партнеров и конкурентов(по 2-3 штуки с учетом потенциала разработанной идеи).

Будь конкретным и практичным. Ответ структурируй по разделам.""",
        
        "web_parser": """Ты - аналитик данных. Проанализируй содержимое веб-страницы и извлеки информацию о бизнес-трендах, нишах и возможностях.
Извлеки: тренды, перспективные ниши, статистику, советы для стартапов, региональные особенности, конкурентные компании и возможные спонсоры.""",
        
        "validator": """Ты - валидатор данных. Проверь, соответствует ли информация критериям:
1. Актуальность (не старше 2 лет)
2. Достоверность (есть ли источники)
3. Релевантность бизнес-тематике
4. Конкретность (есть цифры, факты)
5. Практическая применимость

Верни JSON: {"is_valid": bool, "issues": list, "confidence_score": float}"""
    }

config = Config()