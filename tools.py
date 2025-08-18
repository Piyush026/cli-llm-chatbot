import math
import os
import httpx
import wikipedia
from rich import print
from dotenv import load_dotenv

load_dotenv()
WEATHER_API_KEY = os.environ["WEATHER_API_KEY"]


async def get_weather(city: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.tomorrow.io/v4/weather/realtime?location={city}&apikey={WEATHER_API_KEY}")
        data = response.json()
        print(data)
        if data.get("data"):
            temp = data["data"]["values"].get("temperature", "No alerts")
            return f"The current temperature in {city} is {temp}°C ."
        return "No active alerts"


def search_wikipedia(query: str):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception as e:
        return f"(wikipedia error) {str(e)}"


def calculate(expression: str):
    try:
        result = eval(expression, {"__builtins__": None, "math": math})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"(calculation error) {str(e)}"