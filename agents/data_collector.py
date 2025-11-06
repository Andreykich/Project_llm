from .base_agent import BaseAgent
from config import config
import json

class DataCollectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Data Collector",
            system_prompt=config.PROMPTS["data_collector"]
        )
        self.conversation_history = []
    
    def start_conversation(self) -> str:
        """Начинает диалог с пользователем"""
        return "Привет! Я помогу тебе с бизнес-идеей. Расскажи, какую сферу деятельности ты рассматриваешь?"
    
    def process_user_input(self, user_input: str) -> tuple:
        """
        Обрабатывает ввод пользователя
        Возвращает (следующий_вопрос, собранные_данные или None)
        """
        self.conversation_history.append(f"Пользователь: {user_input}")
        
        prompt = f"""
        История диалога:
        {'\n'.join(self.conversation_history[-6:])}  # Последние 6 сообщений
        
        На основе этого диалога:
        1. Определи, какая информация уже собрана
        2. Реши, нужно ли задать ещё вопрос для получения всей информации
        3. Если вся информация собрана - верни JSON с данными
        4. Если нет - задай следующий уточняющий вопрос
        
        Структура JSON для полных данных:
        {{
            "industry": "сфера деятельности",
            "idea": "конкретная бизнес-идея", 
            "city": "город",
            "budget": "бюджет",
            "experience": "опыт в сфере",
            "target_audience": "целевая аудитория",
            "special_requirements": "особые пожелания"
        }}
        """
        
        response = self.call_llm(prompt, temperature=0.3)
        self.conversation_history.append(f"Ассистент: {response}")
        
        # Пытаемся извлечь JSON (значит данные собраны)
        try:
            data = self.extract_json(response)
            if all(key in data for key in ["industry", "idea", "city"]):
                return None, data  # Данные собраны
        except:
            pass
        
        return response, None  # Продолжаем диалог