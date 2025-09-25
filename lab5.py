# Lab5.py — “What to Wear” Weather Bot (per Lab 5 instructions)
# - Single-shot app: user enters a city, bot returns clothing + picnic advice
# - Uses OpenWeatherMap API (key from st.secrets)
# - Implements OpenAI tool/function: get_current_weather(location) with tool_choice="auto"
# - Properly appends the assistant message with `tool_calls` BEFORE any role="tool" messages
# - Streams the final answer

import streamlit as st
import requests
from openai import OpenAI
import json
from typing import Dict, Any

st.set_page_config(page_title="Lab 5 — What to Wear Bot", page_icon="🌤️")
st.title("🌤️ Lab 5 — What to Wear (Weather + Clothing + Picnic Advice)")

# --- Secrets ---
def _secret(name: str) -> str:
    try:
        return st.secrets[name].strip().replace("\r", "").replace("\n", "")
    except KeyError:
        return ""

OPENAI_API_KEY = _secret("OPENAI_API_KEY")
OWM_API_KEY    = _secret("OPENWEATHER_API_KEY")  # OpenWeatherMap key

if not OPENAI_API_KEY:
    st.error("Missing `OPENAI_API_KEY` in Streamlit secrets.")
    st.stop()
if not OWM_API_KEY:
    st.error("Missing `OPENWEATHER_API_KEY` in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)

# --- Weather tool implementation (Python) ---
def get_current_weather(location: str, api_key: str) -> Dict[str, Any]:
    """
    Calls OpenWeatherMap Current Weather endpoint by city name.
    Extracts temperature (C/F), feels_like, min, max, humidity, wind, and condition.
    If the location contains a comma, we take the city name and country name.
    """
    #if "," in location:
        #location = location.split(",")[0].strip()

    base = "https://api.openweathermap.org/data/2.5/"
    url = f"{base}weather?q={location}&appid={api_key}"
    resp = requests.get(url, timeout=20)
    data = resp.json()

    if resp.status_code != 200 or "main" not in data:
        return {
            "location": location,
            "error": data.get("message", f"Failed to fetch weather for '{location}'"),
        }

    # Kelvin → Celsius / Fahrenheit
    def k2c(k): return round(k - 273.15, 2)
    def k2f(k): return round((k - 273.15) * 9/5 + 32, 2)

    main = data.get("main", {})
    weather_list = data.get("weather", [])
    weather_desc = weather_list[0].get("description") if weather_list else "unknown"
    wind = data.get("wind", {})

    temp_k = main.get("temp")
    feels_k = main.get("feels_like")
    min_k = main.get("temp_min")
    max_k = main.get("temp_max")

    result = {
        "location": location,
        "description": weather_desc,
        "temperature_c": k2c(temp_k) if temp_k is not None else None,
        "temperature_f": k2f(temp_k) if temp_k is not None else None,
        "feels_like_c": k2c(feels_k) if feels_k is not None else None,
        "feels_like_f": k2f(feels_k) if feels_k is not None else None,
        "temp_min_c": k2c(min_k) if min_k is not None else None,
        "temp_max_c": k2c(max_k) if max_k is not None else None,
        "humidity": main.get("humidity"),
        "wind_speed_mps": wind.get("speed"),
        "wind_deg": wind.get("deg"),
    }
    return result

# --- UI: simple form ---
with st.form("weather_form"):
    st.write("Enter a city and I’ll suggest what to wear and whether it’s a good day for a picnic.")
    city = st.text_input("City", placeholder="e.g., Syracuse, US", value="")
    submitted = st.form_submit_button("Get suggestion")

if submitted:
    user_location = city.strip() if city.strip() else "Syracuse, US"

    # --- Messages + Tool definition (OpenAI function calling) ---
    SYSTEM_PROMPT = (
        "You are a helpful travel/clothing assistant. "
        "If a weather tool is provided, call it to get current weather for the requested location. "
        "After you have the weather data, give: "
        "1) Brief weather summary, "
        "2) What to wear (layers, shoes, accessories), "
        "3) Whether it’s a good day for a picnic and why (consider precip, wind, temp). "
        "Be concise and practical."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a city. Returns temperature, humidity, wind, and conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., 'Syracuse, NY' or 'London'",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Please advise what to wear and picnic suitability for: '{user_location}'. "
                "If you need weather, call the weather tool."
            ),
        },
    ]

    st.subheader("🧠 Reasoning & Tools")
    st.caption("Step 1: Ask the model; let it call the weather tool automatically if needed…")

    first = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=initial_messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )

    # Collect tool calls (if any) from the FIRST assistant message
    tool_outputs = []
    assistant_msg = first.choices[0].message  # the assistant message that may contain tool_calls
    tcalls = getattr(assistant_msg, "tool_calls", None)

    if tcalls:
        # For each requested tool call, run our local function and prepare role="tool" messages
        for call in tcalls:
            fn = call.function
            if fn and fn.name == "get_current_weather":
                try:
                    args = json.loads(fn.arguments or "{}")
                    loc = args.get("location") or user_location
                except Exception:
                    loc = user_location

                st.caption(f"→ Model requested weather for: {loc}")
                weather = get_current_weather(loc, OWM_API_KEY)
                tool_outputs.append(
                    {
                        "id": call.id,        # tie back to the tool call id
                        "name": fn.name,
                        "content": json.dumps(weather),
                    }
                )

    # Build the follow-up conversation:
    # (1) system + user (initial_messages)
    followup_messages = list(initial_messages)
    # (2) append the assistant message that contained the tool_calls (REQUIRED)
    #     even if its content is empty, include it with tool_calls so role="tool" is valid next.
    followup_messages.append({
        "role": assistant_msg.role,
        "content": assistant_msg.content,
        "tool_calls": assistant_msg.tool_calls
    })

    if tool_outputs:
        # (3) append our tool results (role="tool") bound to the tool_call_id
        for out in tool_outputs:
            followup_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": out["id"],
                    "name": out["name"],
                    "content": out["content"],
                }
            )
    else:
        # No tool call requested; provide a fallback weather fetch in a system message
        st.caption("→ No tool call detected; using fallback weather fetch.")
        w = get_current_weather(user_location, OWM_API_KEY)
        followup_messages.append(
            {"role": "system", "content": f"Weather data (fallback): {json.dumps(w)}"}
        )

    # Step 2: stream the final suggestion
    st.subheader("👚 Suggestion")
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=followup_messages,
            stream=True,
            temperature=0.3,
        )
        final_text = st.write_stream(stream)

    # Optional: Show the raw weather JSON
    with st.expander("Raw weather data"):
        if tool_outputs:
            try:
                st.json(json.loads(tool_outputs[0]["content"]))
            except Exception:
                st.write(tool_outputs[0]["content"])
        else:
            st.write("Used fallback fetch above.")
