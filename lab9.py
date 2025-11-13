# streamlit_app.py

from openai import OpenAI
import streamlit as st
import requests
import datetime as dt
import json
from typing import Dict, Any, Optional

# ============================================================
# Secrets and clients
# ============================================================

# Show which secrets exist (for debugging if needed)
# You can comment this out once things work.
st.write("Available secrets keys:", list(st.secrets.keys()))

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
OPENWEATHERMAP_API_KEY = st.secrets["OPENWEATHERMAP_API_KEY"]

client = OpenAI(api_key=OPENAI_KEY)

# ============================================================
# Helper: LLM call using new OpenAI client
# ============================================================

def llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    """Wrapper for OpenAI chat completions using the v1 client."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# Helper: Weather fetching and summarizing
# ============================================================

def get_weather_data(city: str) -> Optional[Dict[str, Any]]:
    """Fetch 5-day / 3-hour forecast from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric",
    }
    try:
        res = requests.get(url, params=params, timeout=10)
    except requests.RequestException:
        return None

    if res.status_code != 200:
        return None

    data = res.json()
    if "list" not in data or "city" not in data:
        return None

    return data


def summarize_weather_for_date(
    weather_data: Dict[str, Any],
    target_date: dt.date,
) -> Dict[str, Any]:
    """
    Given OpenWeatherMap forecast data and a target date,
    compute a simple summary for that date (min/max temp, condition).
    """
    entries = weather_data.get("list", [])
    if not entries:
        return {}

    temps = []
    conditions = []

    for entry in entries:
        ts = entry.get("dt")
        if ts is None:
            continue
        t = dt.datetime.utcfromtimestamp(ts)
        entry_date = t.date()

        if entry_date == target_date:
            main = entry.get("main", {})
            weather_list = entry.get("weather", [])
            temp = main.get("temp")
            if temp is not None:
                temps.append(temp)
            if weather_list:
                conditions.append(weather_list[0].get("description", ""))

    # If nothing matches that exact date, fall back to all entries
    fallback = False
    if not temps:
        fallback = True
        for entry in entries:
            main = entry.get("main", {})
            weather_list = entry.get("weather", [])
            temp = main.get("temp")
            if temp is not None:
                temps.append(temp)
            if weather_list:
                conditions.append(weather_list[0].get("description", ""))

    if not temps:
        return {}

    avg_temp = sum(temps) / len(temps)
    min_temp = min(temps)
    max_temp = max(temps)

    if conditions:
        most_common_condition = max(set(conditions), key=conditions.count)
    else:
        most_common_condition = "unknown"

    return {
        "avg_temp": avg_temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "condition": most_common_condition,
        "city": weather_data.get("city", {}).get("name", ""),
        "fallback": fallback,
    }


def llm_weather_prediction(
    city: str,
    base_summary: Dict[str, Any],
    days_ahead: int,
) -> str:
    """Use the LLM to describe likely weather trends for dates > 2 days out."""
    system_prompt = (
        "You are a weather reasoning assistant. "
        "Given a rough forecast summary and days ahead, "
        "describe a realistic but approximate expectation of the weather. "
        "Explicitly note that it is a rough prediction."
    )

    user_prompt = f"""
City: {city}
Days ahead: {days_ahead}

Base forecast summary (from 5-day forecast):
- Average temp: {base_summary.get('avg_temp', 'unknown')} °C
- Min temp: {base_summary.get('min_temp', 'unknown')} °C
- Max temp: {base_summary.get('max_temp', 'unknown')} °C
- Common condition: {base_summary.get('condition', 'unknown')}

Explain expected trends and uncertainty.
"""

    return llm_call(system_prompt, user_prompt)

# ============================================================
# Helper: Travel info from LLM
# ============================================================

def calculate_travel_info(origin: str, destination: str) -> Dict[str, Any]:
    """
    Use the LLM to estimate distance and travel time between origin and
    destination and suggest reasonable travel modes.
    """
    system_prompt = (
        "You are a travel logistics expert. Estimate realistic travel distances "
        "and times between places using general knowledge (no maps API)."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}

Tasks:
1. Estimate approximate distance between these locations.
2. Estimate typical travel time by car (if feasible).
3. Estimate typical travel time by air (if typical flight routes exist).
4. Recommend if driving, flying, train, or other modes make most sense.
Output your answer in short bullet points.
"""

    text = llm_call(system_prompt, user_prompt)
    return {"summary": text}

# ============================================================
# Agents
# ============================================================

def weather_agent(
    origin: str,
    destination: str,
    departure_date: dt.date,
    origin_weather: Dict[str, Any],
    dest_weather: Dict[str, Any],
    days_ahead: int,
) -> str:
    """Compare weather at origin and destination on the trip date."""
    origin_summary = summarize_weather_for_date(origin_weather, departure_date)
    dest_summary = summarize_weather_for_date(dest_weather, departure_date)

    origin_pred = ""
    dest_pred = ""

    if days_ahead > 2:
        if origin_summary:
            origin_pred = llm_weather_prediction(
                origin_summary.get("city", origin),
                origin_summary,
                days_ahead,
            )
        if dest_summary:
            dest_pred = llm_weather_prediction(
                dest_summary.get("city", destination),
                dest_summary,
                days_ahead,
            )

    system_prompt = (
        "You are a travel weather advisor. Compare the weather at origin and "
        "destination on the chosen trip date. Consider temperatures, conditions, "
        "and any provided predictions."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}
Departure date: {departure_date.isoformat()}
Days ahead: {days_ahead}

Origin weather summary:
{json.dumps(origin_summary, indent=2)}

Destination weather summary:
{json.dumps(dest_summary, indent=2)}

Origin prediction (may be empty):
{origin_pred}

Destination prediction (may be empty):
{dest_pred}

Provide:
- A concise comparison of origin vs destination weather.
- Any notable risks (storms, heat, cold, heavy rain).
- How confident the forecast is, especially if days_ahead > 2.
"""

    return llm_call(system_prompt, user_prompt)


def logistics_agent(
    origin: str,
    destination: str,
    trip_duration_days: int,
    weather_summary_text: str,
    travel_info: Dict[str, Any],
) -> str:
    """Recommend travel mode, timing, and logistic tips."""
    system_prompt = (
        "You are a travel logistics planner. Use weather, trip length, and "
        "rough travel estimates to give practical advice."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}
Trip length: {trip_duration_days} days

Weather summary:
{weather_summary_text}

Estimated travel info:
{travel_info.get('summary', '')}

Tasks:
- Recommend the best primary travel mode (car, plane, train, etc.).
- Suggest approximate ideal departure and return timing.
- Call out constraints (international borders, oceans, long distances).
- Include 3 to 5 practical tips (tickets, check-in, connections, local transit).
- Never suggest driving across oceans or between continents.
"""

    return llm_call(system_prompt, user_prompt)


def packing_agent(
    destination: str,
    trip_duration_days: int,
    weather_summary_text: str,
) -> str:
    """Suggest clothing and accessories for the weather and trip length."""
    system_prompt = (
        "You are a packing assistant. Suggest what to pack based on "
        "destination weather and trip duration. Assume a typical leisure trip."
    )

    user_prompt = f"""
Destination: {destination}
Trip length: {trip_duration_days} days

Weather summary:
{weather_summary_text}

Tasks:
- List clothing in grouped categories (tops, bottoms, outerwear, footwear).
- Include weather-dependent items (umbrella, hat, gloves, sunglasses, etc.).
- List essentials (toiletries, documents, electronics).
- Assume the traveler prefers to pack efficiently and re-wear some items.
Format as bullet lists.
"""

    return llm_call(system_prompt, user_prompt)


def activity_agent(
    destination: str,
    trip_duration_days: int,
    weather_summary_text: str,
) -> str:
    """Create a day-by-day itinerary with local suggestions."""
    system_prompt = (
        "You are a travel activity planner. Create realistic day-by-day "
        "itineraries using weather information."
    )

    user_prompt = f"""
Destination: {destination}
Trip length: {trip_duration_days} days

Weather summary:
{weather_summary_text}

Tasks:
- Create a day-by-day itinerary (Day 1, Day 2, etc.).
- For each day include morning, afternoon, and evening suggestions.
- Balance indoor and outdoor options based on the weather.
- Include at least one food or café idea each day.
Keep the suggestions general enough to apply to most travelers.
"""

    return llm_call(system_prompt, user_prompt)


def overview_agent(
    origin: str,
    destination: str,
    departure_date: dt.date,
    trip_duration_days: int,
    weather_text: str,
    logistics_text: str,
    packing_text: str,
    itinerary_text: str,
) -> str:
    """High-level overview of the entire trip plan."""
    system_prompt = (
        "You are a travel summary assistant. Summarize the overall trip "
        "in a clear, short overview."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}
Departure date: {departure_date.isoformat()}
Trip length: {trip_duration_days} days

Weather section excerpt:
{weather_text[:600]}

Logistics section excerpt:
{logistics_text[:400]}

Packing section excerpt:
{packing_text[:400]}

Itinerary section excerpt:
{itinerary_text[:400]}

Provide a short overview (1–3 paragraphs) that:
- Summarizes the trip style and purpose.
- Highlights key weather expectations.
- Mentions the main travel mode and the overall pace of the itinerary.
"""

    return llm_call(system_prompt, user_prompt)

# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title="Multi-Agent Travel Planner", layout="wide")
    st.title("Lab 9 – Multi-Agent Travel Planning System")

    st.markdown(
        "This app uses OpenWeatherMap for weather data and OpenAI for four agents: "
        "Weather, Logistics, Packing, and Activity."
    )

    with st.sidebar:
        st.header("Trip Inputs")

        origin = st.text_input("Origin city", value="Syracuse, NY")
        destination = st.text_input("Destination city", value="Miami, FL")

        today = dt.date.today()
        departure_date = st.date_input(
            "Departure date",
            value=today + dt.timedelta(days=3),
            min_value=today,
        )

        trip_duration_days = st.number_input(
            "Trip duration (days)",
            min_value=1,
            max_value=30,
            value=5,
        )

        run_button = st.button("Plan Trip")

    if not run_button:
        st.info("Enter your trip details and click Plan Trip.")
        return

    if not origin or not destination:
        st.error("Please provide both origin and destination.")
        return

    days_ahead = (departure_date - dt.date.today()).days

    # Weather data
    with st.spinner("Fetching weather data..."):
        origin_weather = get_weather_data(origin)
        dest_weather = get_weather_data(destination)

    if origin_weather is None:
        st.error(f"Could not fetch weather data for origin: {origin}")
        return
    if dest_weather is None:
        st.error(f"Could not fetch weather data for destination: {destination}")
        return

    # Travel info
    with st.spinner("Estimating travel logistics..."):
        travel_info = calculate_travel_info(origin, destination)

    # Agents
    with st.spinner("Running Weather Agent..."):
        weather_text = weather_agent(
            origin,
            destination,
            departure_date,
            origin_weather,
            dest_weather,
            days_ahead,
        )

    with st.spinner("Running Logistics Agent..."):
        logistics_text = logistics_agent(
            origin,
            destination,
            trip_duration_days,
            weather_text,
            travel_info,
        )

    with st.spinner("Running Packing Agent..."):
        packing_text = packing_agent(
            destination,
            trip_duration_days,
            weather_text,
        )

    with st.spinner("Running Activity Agent..."):
        itinerary_text = activity_agent(
            destination,
            trip_duration_days,
            weather_text,
        )

    with st.spinner("Generating overview..."):
        overview_text = overview_agent(
            origin,
            destination,
            departure_date,
            trip_duration_days,
            weather_text,
            logistics_text,
            packing_text,
            itinerary_text,
        )

    # Display
    st.subheader("Overview")
    st.write(overview_text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weather")
        st.write(weather_text)

        st.subheader("Logistics")
        st.write(logistics_text)

    with col2:
        st.subheader("Packing")
        st.write(packing_text)

        st.subheader("Itinerary")
        st.write(itinerary_text)

    st.markdown("---")
    st.caption(
        "Implements a simple multi-agent travel planner using OpenWeatherMap and OpenAI."
    )

if __name__ == "__main__":
    main()
