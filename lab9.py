import os
import json
import datetime as dt
from typing import Dict, Any, Optional, Tuple

import requests
import streamlit as st
import openai

# =========================
# Setup: API Keys
# =========================

from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def llm_call(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message["content"]
# =========================
# Helper Functions
# =========================

def llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    """
    Wrapper for OpenAI chat completion calls.
    """
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["choices"][0]["message"]["content"].strip()


def get_weather_data(city: str) -> Optional[Dict[str, Any]]:
    """
    Fetch 5-day / 3-hour forecast from OpenWeatherMap for a given city.
    Returns None if city is invalid or request fails.
    """
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
    # Basic sanity check
    if "list" not in data or "city" not in data:
        return None
    return data


def summarize_weather_for_date(
    weather_data: Dict[str, Any],
    target_date: dt.date,
) -> Dict[str, Any]:
    """
    Given OpenWeatherMap forecast data and a target date,
    compute a simple summary for that date (min/max temp, common condition).
    If no exact date entries exist, it will approximate based on closest times.
    """
    entries = weather_data.get("list", [])
    if not entries:
        return {}

    temps = []
    conditions = []

    for entry in entries:
        ts = entry.get("dt", None)
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

    # If we have no entries on that exact date, fall back to overall forecast summary
    if not temps:
        for entry in entries:
            main = entry.get("main", {})
            weather_list = entry.get("weather", [])
            temp = main.get("temp")
            if temp is not None:
                temps.append(temp)
            if weather_list:
                conditions.append(weather_list[0].get("description", ""))
        fallback = True
    else:
        fallback = False

    if not temps:
        return {}

    avg_temp = sum(temps) / len(temps)
    min_temp = min(temps)
    max_temp = max(temps)

    # Simple mode for conditions
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
    """
    Use the LLM to reason about likely weather trends if the trip is more
    than 2 days ahead.
    """
    system_prompt = (
        "You are a weather reasoning assistant. You receive a rough forecast "
        "summary and a number of days ahead and you describe a realistic but "
        "clearly approximate expectation of the weather, including uncertainty."
    )

    user_prompt = f"""
City: {city}
Days ahead: {days_ahead}
Base forecast summary (from 5 day forecast):
- Average temp: {base_summary.get('avg_temp', 'unknown')} °C
- Min temp: {base_summary.get('min_temp', 'unknown')} °C
- Max temp: {base_summary.get('max_temp', 'unknown')} °C
- Common condition: {base_summary.get('condition', 'unknown')}

Explain likely weather trends for that date. Make it clear this is a rough prediction.
"""
    return llm_call(system_prompt, user_prompt)


def calculate_travel_info(origin: str, destination: str) -> Dict[str, Any]:
    """
    Use the LLM to estimate distance and travel time between origin and
    destination and suggest reasonable travel modes.
    """
    system_prompt = (
        "You are a travel logistics expert. Estimate realistic travel distances "
        "and times between places using common knowledge, not exact mapping APIs."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}

Tasks:
1. Estimate approximate distance between these locations.
2. Estimate typical travel time by car (if feasible).
3. Estimate typical travel time by air (if common routes exist).
4. Recommend if driving, flying, train, or other modes make most sense.
Return a concise bullet-style explanation.
"""

    text = llm_call(system_prompt, user_prompt)
    return {"summary": text}


# =========================
# Agents
# =========================

def weather_agent(
    origin: str,
    destination: str,
    departure_date: dt.date,
    origin_weather: Dict[str, Any],
    dest_weather: Dict[str, Any],
    days_ahead: int,
) -> str:
    """
    Compare weather at origin and destination, including LLM-based
    prediction if the trip is more than 2 days away.
    """
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
        "destination on the selected trip date. Consider temperatures, conditions, "
        "and predictions if provided."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}
Departure date: {departure_date.isoformat()}
Days ahead: {days_ahead}

Origin weather summary: {json.dumps(origin_summary, indent=2)}
Destination weather summary: {json.dumps(dest_summary, indent=2)}

Origin trend prediction (may be empty): {origin_pred}
Destination trend prediction (may be empty): {dest_pred}

Provide:
- A clear comparison of origin vs destination weather.
- Any important risks (storms, heat, cold, rain).
- How confident the forecast is (especially if days ahead > 2).
"""

    return llm_call(system_prompt, user_prompt)


def logistics_agent(
    origin: str,
    destination: str,
    trip_duration_days: int,
    weather_summary_text: str,
    travel_info: Dict[str, Any],
) -> str:
    """
    Recommend travel mode, timing, and logistics tips.
    """
    system_prompt = (
        "You are a travel logistics planner. You consider weather, trip length, "
        "and rough travel estimates to give practical advice."
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
- Recommend the best primary travel mode (car, plane, train, bus, etc.).
- Advise on ideal departure and return timing.
- Note any constraints (international border, long ocean crossings).
- Include 3 to 5 practical logistics tips (tickets, check-in, connections).
- Make sure you do not suggest driving across oceans or between continents.
"""

    return llm_call(system_prompt, user_prompt)


def packing_agent(
    destination: str,
    trip_duration_days: int,
    weather_summary_text: str,
) -> str:
    """
    Suggest clothing and accessories for the weather and trip length.
    """
    system_prompt = (
        "You are a packing assistant. Suggest what to pack based on weather "
        "and trip duration. Assume a typical leisure trip unless otherwise stated."
    )

    user_prompt = f"""
Destination: {destination}
Trip length: {trip_duration_days} days

Weather summary:
{weather_summary_text}

Tasks:
- List clothing in categories: tops, bottoms, outerwear, footwear.
- Include weather dependent items (umbrella, sun hat, gloves, etc.).
- Include essential accessories and documents.
- Assume the user prefers to pack efficiently with some ability to re-wear items.
Format as concise bullet lists.
"""

    return llm_call(system_prompt, user_prompt)


def activity_agent(
    destination: str,
    trip_duration_days: int,
    weather_summary_text: str,
) -> str:
    """
    Create a day-wise itinerary with local suggestions, taking
    weather into account.
    """
    system_prompt = (
        "You are a travel activity planner. Propose realistic day-by-day "
        "itineraries based on destination and weather."
    )

    user_prompt = f"""
Destination: {destination}
Trip length: {trip_duration_days} days

Weather summary:
{weather_summary_text}

Tasks:
- Create a day-by-day itinerary (Day 1, Day 2, etc.).
- For each day suggest morning, afternoon, and evening ideas.
- Balance indoor and outdoor activities based on the weather.
- Include at least one food or café suggestion each day.
- Keep recommendations generic enough to apply to many travelers.
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
    """
    High-level trip overview that ties everything together.
    """
    system_prompt = (
        "You are a travel summary assistant. Summarize the plan clearly "
        "for the user in a short overview."
    )

    user_prompt = f"""
Origin: {origin}
Destination: {destination}
Departure date: {departure_date.isoformat()}
Trip length: {trip_duration_days} days

Weather section (excerpt):
{weather_text[:1000]}

Logistics section (excerpt):
{logistics_text[:800]}

Packing section (excerpt):
{packing_text[:800]}

Itinerary section (excerpt):
{itinerary_text[:800]}

Provide a short overview (1 to 3 paragraphs) that:
- Summarizes the trip purpose and nature.
- Highlights key weather expectations.
- Mentions the main travel mode and overall rhythm of the itinerary.
"""

    return llm_call(system_prompt, user_prompt)


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(page_title="Multi-Agent Travel Planner", layout="wide")
    st.title("Lab 9 – Multi-Agent Travel Planning System")

    st.markdown(
        "This app uses OpenWeatherMap for weather data and OpenAI for four agents: "
        "Weather, Logistics, Packing, and Activities."
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
        st.info("Enter your trip details in the sidebar and click Plan Trip.")
        return

    if not origin or not destination:
        st.error("Please provide both origin and destination.")
        return

    # Compute days ahead
    days_ahead = (departure_date - dt.date.today()).days

    # Fetch weather data
    with st.spinner("Fetching weather data..."):
        origin_weather = get_weather_data(origin)
        dest_weather = get_weather_data(destination)

    if origin_weather is None:
        st.error(f"Could not fetch weather data for origin: {origin}")
        return
    if dest_weather is None:
        st.error(f"Could not fetch weather data for destination: {destination}")
        return

    # Travel info from LLM
    with st.spinner("Estimating travel logistics..."):
        travel_info = calculate_travel_info(origin, destination)

    # Weather agent
    with st.spinner("Running Weather Agent..."):
        weather_text = weather_agent(
            origin,
            destination,
            departure_date,
            origin_weather,
            dest_weather,
            days_ahead,
        )

    # Logistics agent
    with st.spinner("Running Logistics Agent..."):
        logistics_text = logistics_agent(
            origin,
            destination,
            trip_duration_days,
            weather_text,
            travel_info,
        )

    # Packing agent
    with st.spinner("Running Packing Agent..."):
        packing_text = packing_agent(
            destination,
            trip_duration_days,
            weather_text,
        )

    # Activity agent
    with st.spinner("Running Activity Agent..."):
        itinerary_text = activity_agent(
            destination,
            trip_duration_days,
            weather_text,
        )

    # Overview agent
    with st.spinner("Generating trip overview..."):
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

    # =========================
    # Display Results
    # =========================

    st.subheader("Overview")
    st.write(overview_text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weather Comparison")
        st.write(weather_text)

        st.subheader("Logistics")
        st.write(logistics_text)

    with col2:
        st.subheader("Packing List")
        st.write(packing_text)

        st.subheader("Day-by-Day Itinerary")
        st.write(itinerary_text)

    st.markdown("---")
    st.markdown(
        "This output demonstrates a simple multi-agent system using a single LLM provider (OpenAI) "
        "plus OpenWeatherMap for external data."
    )


if __name__ == "__main__":
    main()
