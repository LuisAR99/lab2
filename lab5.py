import requests
def get_current_weather(location, API_key):
if "," in location:
location = location.split(",")[0].strip()
urlbase = "https://api.openweathermap.org/data/2.5/"
urlweather = f"weather?q={location}&appid={API_key}"
url = urlbase + urlweather
response = requests.get(url)
data = response.json()
# Extract temperatures & Convert Kelvin to Celsius
temp = data['main']['temp'] - 273.15
feels_like = data['main']['feels_like'] - 273.15
temp_min = data['main']['temp_min'] - 273.15
temp_max = data['main']['temp_max'] - 273.15
humidity = data['main']['humidity']
return {"location": location,
"temperature": round(temp, 2),
"feels_like": round(feels_like, 2),
"temp_min": round(temp_min, 2),
"temp_max": round(temp_max, 2),
"humidity": round(humidity, 2)
}
