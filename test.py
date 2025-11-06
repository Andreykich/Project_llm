# test_deepseek.py
from agents.base_agent import BaseAgent
import config

def test_deepseek():
    agent = BaseAgent("test", "Ты полезный ассистент")
    response = agent.call_llm("Привет! Ответь коротко: как дела?")
    print("DeepSeek ответ:", response)

if __name__ == "__main__":
    test_deepseek()

