from openai import OpenAI
from config import config
import json
import re
from typing import Callable, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Базовый класс для агентов с поддержкой Function Calling.
    
    Улучшения (на основе Лекции 9 - Function Calling & MCP):
    - Function Calling: LLM может самостоятельно вызывать инструменты
    - Явный контроль над инструментами
    - Валидация структурированного вывода
    - Рассуждения в стиле Chain-of-Thought
    """
    
    def __init__(self, name: str, system_prompt: str, tools: Optional[List[Dict[str, Any]]] = None):
        self.name = name
        self.client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL
        )
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.thinking_process = []  # Для отслеживания процесса мышления
        
        logger.info(f"Инициализирован агент: {name}")
    
    def register_tool(self, tool_name: str, tool_func: Callable, description: str, parameters: Dict[str, Any]):
        """
        Регистрирует инструмент для использования LLM.
        
        Args:
            tool_name: Имя инструмента (для вызова)
            tool_func: Python-функция, которая выполняет инструмент
            description: Описание для LLM — что делает инструмент
            parameters: JSON Schema для параметров
        """
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools.append(tool_definition)
        
        # Сохраняем маппинг: имя → функция
        if not hasattr(self, '_tool_implementations'):
            self._tool_implementations = {}
        self._tool_implementations[tool_name] = tool_func
        
        logger.info(f"Инструмент '{tool_name}' зарегистрирован в агенте {self.name}")
    
    def call_llm_with_tools(self, prompt: str, temperature: float = None, max_retries: int = 3) -> Dict[str, Any]:
        """
        Вызывает LLM с поддержкой Function Calling.
        
        Процесс (на основе Лекции 9, слайды 5-6):
        1. Отправляем промпт + определения инструментов
        2. LLM решает, какие инструменты нужны
        3. Мы выполняем эти инструменты
        4. Отправляем результаты обратно в LLM
        5. LLM генерирует финальный ответ
        
        Args:
            prompt: Основной промпт для LLM
            temperature: Температура генерации
            max_retries: Максимальное количество повторных попыток
            
        Returns:
            Dict с результатом: {'response': str, 'tool_calls': list, 'reasoning': str}
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Первый вызов LLM (может выбрать инструменты)
            logger.info(f"[{self.name}] Первый вызов LLM...")
            response = self.client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=temperature or config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None
            )
            
            self.thinking_process.append(f"LLM выбрал: {response.choices[0].finish_reason}")
            
            result = {
                "response": "",
                "tool_calls": [],
                "reasoning": "",
                "raw_response": response
            }
            
            # Обработка tool_calls, если LLM их выбрал
            if response.choices[0].finish_reason == "tool_calls" and response.choices[0].message.tool_calls:
                logger.info(f"[{self.name}] LLM выбрал инструменты. Обработка...")
                
                tool_calls = response.choices[0].message.tool_calls
                tool_results = []
                
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"[{self.name}] Вызов инструмента: {tool_name} с args: {tool_args}")
                    
                    # Выполняем инструмент
                    if hasattr(self, '_tool_implementations') and tool_name in self._tool_implementations:
                        try:
                            tool_result = self._tool_implementations[tool_name](**tool_args)
                            tool_results.append({
                                "tool_call_id": tool_call.id,
                                "result": tool_result,
                                "success": True
                            })
                            result["tool_calls"].append({
                                "name": tool_name,
                                "args": tool_args,
                                "result": tool_result
                            })
                        except Exception as e:
                            logger.error(f"Ошибка при выполнении инструмента {tool_name}: {e}")
                            tool_results.append({
                                "tool_call_id": tool_call.id,
                                "result": f"Ошибка: {str(e)}",
                                "success": False
                            })
                    else:
                        logger.warning(f"Инструмент {tool_name} не найден")
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "result": f"Инструмент {tool_name} недоступен",
                            "success": False
                        })
                
                # Второй вызов LLM с результатами инструментов
                logger.info(f"[{self.name}] Второй вызов LLM с результатами инструментов...")
                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                
                for tool_result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_result["tool_call_id"],
                        "content": json.dumps(tool_result["result"])
                    })
                
                # Финальный вызов для получения ответа
                final_response = self.client.chat.completions.create(
                    model=config.MODEL,
                    messages=messages,
                    temperature=temperature or config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                
                result["response"] = final_response.choices[0].message.content
                result["reasoning"] = f"Использовано {len(tool_calls)} инструментов для анализа"
                
            else:
                # Нет tool_calls — просто текстовый ответ
                result["response"] = response.choices[0].message.content
                result["reasoning"] = "Прямой ответ без использования инструментов"
            
            logger.info(f"[{self.name}] Успешный вызов LLM")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при вызове LLM в {self.name}: {e}")
            return {
                "response": f"Ошибка: {str(e)}",
                "tool_calls": [],
                "reasoning": "Обработка ошибки",
                "error": str(e)
            }
    
    def call_llm(self, prompt: str, temperature: float = None) -> str:
        """
        Обратная совместимость: простой вызов LLM без инструментов.
        """
        result = self.call_llm_with_tools(prompt, temperature)
        return result["response"]
    
    def extract_json(self, text: str) -> dict:
        """Извлекает JSON из текста ответа"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"raw_response": text}
        except:
            return {"raw_response": text}
    
    def add_reasoning_step(self, step: str):
        """Добавляет шаг в процесс мышления (для Chain-of-Thought)"""
        self.thinking_process.append(step)
        logger.info(f"[{self.name}] Мышление: {step}")
    
    def get_reasoning_trace(self) -> str:
        """Возвращает трассу мышления для отладки"""
        return "\n".join(self.thinking_process)
