import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool
from langchain_community.tools import BraveSearch
from dotenv import load_dotenv
load_dotenv(override=True)

API_KEY = os.getenv("BRAVE_API_KEY")
WEATHER_URL = os.getenv("WEATHER_URL")
WEATHER_ACCESS_TOKEN = os.getenv("WEATHER_ACCESS_TOKEN")

def _default_bkk_hourstamp() -> str:
    # Format: YYYY-MM-DDTHH:00:00 in Asia/Bangkok
    now = datetime.now(ZoneInfo("Asia/Bangkok"))
    return now.strftime("%Y-%m-%dT%H:00:00")

@tool
def get_current_weather(province_th="กรุงเทพมหานคร", amphoe_th="จตุจักร") -> dict:
    """Get weather for a given city."""
    """กรมอุตุ https://data.tmd.go.th/nwpapi/doc/apidoc/forecast_area.html"""

    querystring = {"domain":"2", 
                   "province": province_th, 
                   "amphoe":amphoe_th, 
                   "fields":"tc,rh", 
                   "starttime": _default_bkk_hourstamp()} # Format YYYY-MM-DDTHH:00:00

    headers = {
        'accept': "application/json",
        'authorization': f"Bearer {WEATHER_ACCESS_TOKEN}",
        }

    response = requests.request("GET", 
                                WEATHER_URL, 
                                headers=headers, 
                                params=querystring)

    res = response.json()

    if len(res['WeatherForecasts']) < 1:
        return f"No valid weather information in {amphoe_th}, {province_th}.\nPlease check your location" 
    
    forecast = res['WeatherForecasts'][0]
    return {
        'location': {"lat": forecast['location']['lat'], "lon": forecast['location']['lon']},
        'time': forecast['forecasts'][0]['time'],
        'temperature (°C)': forecast['forecasts'][0]['data']['tc'],
        'relative_humidity (%)': forecast['forecasts'][0]['data']['rh']
        }

# Brave Search Tool
search_tool = BraveSearch.from_api_key(api_key=API_KEY, search_kwargs={"count": 3}) # print(tool.run("What is Today in Bangkok"))

if __name__ == "__main__":
    res = get_current_weather.invoke({"province_th":"กรุงเทพมหานคร", "amphoe_th":"จตุจักร"})
    print(res)
