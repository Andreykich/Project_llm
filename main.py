from agents.data_collector import DataCollectorAgent
from agents.data_analyzer import DataAnalyzerAgent
from agents.web_parser import WebParserAgent
from agents.validator import ValidatorAgent
from database.json_db import JSONDatabase
import json

def main():
    # Инициализация компонентов
    db = JSONDatabase("data/database.json")
    collector = DataCollectorAgent()
    analyzer = DataAnalyzerAgent(db)
    parser = WebParserAgent()
    validator = ValidatorAgent()
    
    print("=== Бизнес-консультант AI ===")
    print(collector.start_conversation())
    
    # Диалог с пользователем
    user_data = None
    while True:
        user_input = input("\nВы: ")
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
            
        next_question, collected_data = collector.process_user_input(user_input)
        
        if collected_data:
            user_data = collected_data
            print("\n✓ Данные собраны! Анализирую...")
            break
        else:
            print(f"Консультант: {next_question}")
    
    if user_data:
        # Генерируем советы
        advice = analyzer.generate_advice(user_data)
        print(f"\n🎯 РЕКОМЕНДАЦИИ ДЛЯ ВАШЕГО БИЗНЕСА:\n")
        print(advice)
        
        # Сохраняем запрос для улучшения системы
        db.add_parsed_source({
            "type": "user_query",
            "data": user_data,
            "response_preview": advice[:200] + "..."
        })

def developer_mode():
    """Режим для разработчиков - парсинг сайтов"""
    db = JSONDatabase("data/database.json")
    parser = WebParserAgent()
    validator = ValidatorAgent()
    
    urls = [
        "https://habr.com/ru/companies/domclick/articles/928600/"
        # Добавьте свои URL здесь
    ]
    
    for url in urls:
        print(f"Парсинг {url}...")
        data = parser.parse_website(url)
        
        # Валидируем данные
        validation = validator.validate_data(data)
        
        if validation.get('is_valid', False):
            # Сохраняем в БД
            for trend in data.get('trends', []):
                db.add_business_trend({
                    "industry": data.get('industry', 'IT'),
                    "trend": trend,
                    "description": f"Извлечено с {url}",
                    "sources": [url],
                    "confidence": validation.get('confidence_score', 0.5)
                })
            print("✓ Данные сохранены")
        else:
            print("✗ Данные не прошли валидацию:", validation.get('issues', []), validation.get('confidence_score'))

if __name__ == "__main__":
    # Режим пользователя
    main()
    
    # Или режим разработчика (раскомментируйте)
    #developer_mode()
    # url ="https://habr.com/ru/companies/domclick/articles/928600/"
    # parser = WebParserAgent()

    # print(parser.parse_website(url))