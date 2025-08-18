import os

import httpx
from rich import print
from dotenv import load_dotenv
load_dotenv()
WEATHER_API_KEY = os.environ["WEATHER_API_KEY"]
async def get_weather(city: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.tomorrow.io/v4/weather/realtime?location={city}&apikey={WEATHER_API_KEY}")
        data = response.json()
        print(data)
        if data.get("data"):
            temp = data["data"]["values"].get("temperature", "No alerts")
            return f"The current temperature in {city} is {temp}°C ."
        return "No active alerts"