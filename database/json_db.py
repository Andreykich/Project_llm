import json
import os
from typing import Dict, List, Any
from datetime import datetime

class JSONDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Создаёт файл БД если его нет"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            initial_data = {
                "business_trends": [],
                "success_stories": [],
                "market_data": [],
                "parsed_sources": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self._save_data(initial_data)
    
    def _load_data(self) -> Dict:
        with open(self.db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_data(self, data: Dict):
        data['updated_at'] = datetime.now().isoformat()
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_business_trend(self, trend_data: Dict):
        data = self._load_data()
        trend_data['id'] = len(data['business_trends']) + 1
        trend_data['created_at'] = datetime.now().isoformat()
        data['business_trends'].append(trend_data)
        self._save_data(data)
    
    def search_trends(self, industry: str, city: str = None) -> List[Dict]:
        data = self._load_data()
        results = []
        
        for trend in data['business_trends']:
            if industry.lower() in trend.get('industry', '').lower():
                if city and trend.get('city'):
                    if city.lower() in trend.get('city', '').lower():
                        results.append(trend)
                else:
                    results.append(trend)
        
        return results
    
    def add_parsed_source(self, source_data: Dict):
        data = self._load_data()
        data['parsed_sources'].append({
            **source_data,
            'parsed_at': datetime.now().isoformat()
        })
        self._save_data(data)

# Пример использования
if __name__ == "__main__":
    db = JSONDatabase("data/database.json")
    
    # Добавляем тестовые данные
    db.add_business_trend({
        "industry": "IT",
        "sub_industry": "Разработка SaaS",
        "trend": "Low-code платформы",
        "description": "Растёт спрос на low-code решения для малого бизнеса",
        "opportunity": "Высокий",
        "risks": "Высокая конкуренция",
        "sources": ["Habr", "VC.ru"],
        "city": "Москва"
    })