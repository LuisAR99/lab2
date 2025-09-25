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
OWM_API_KEY    = _secret("OPENWEATHER_API_KEY")  # e.g., "abc123..." from OpenWeather

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
    Extracts temperature (C/F), feels_like, min, max, humidity, condition, wind, etc.
    If the location contains a comma, we only take the city name (per lab hint).
    """
    if "," in location:
        location = location.split(",")[0].strip()

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

# --- UI: simple form (NOT a chat) ---
with st.form("weather_form"):
    st.write("Enter a city and I’ll suggest what to wear and if it’s a good day for a picnic.")
    city = st.text_input("City", placeholder="e.g., Syracuse, NY", value="")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.form_submit_button("Use Syracuse, NY"):
            city = "Syracuse, NY"
    with col_b:
        if st.form_submit_button("Use London, England"):
            city = "London, England"
    with col_c:
        submit = st.form_submit_button("Get suggestion")

# If either explicit submit or a quick-pick is pressed, run
if submit or city in ("Syracuse, NY", "London, England"):
    user_location = city.strip() if city.strip() else "Syracuse, NY"  # default per lab

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

    # Tool schema for auto function-calling
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

    # First call: allow the model to request the tool if it needs weather
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

    # Check if the model asked to call our tool
    tool_outputs = []
    for choice in first.choices:
        tcalls = getattr(choice.message, "tool_calls", None)
        if not tcalls:
            continue
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
                        "id": call.id,  # tie back to this tool call
                        "name": fn.name,
                        "content": json.dumps(weather),
                    }
                )

    # Build message list for the second call (include any tool outputs)
    followup_messages = list(initial_messages)
    if tool_outputs:
        # Attach each tool result as a message with role=tool
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
        # If the model didn't call the tool (unlikely), add a gentle nudge with a default fetch
        st.caption("→ No tool call detected; using default weather fetch.")
        w = get_current_weather(user_location, OWM_API_KEY)
        followup_messages.append(
            {
                "role": "system",
                "content": f"Weather data (fallback): {json.dumps(w)}",
            }
        )

    # Second call: stream the final suggestion
    st.subheader("👚 Suggestion")
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=followup_messages,
            stream=True,
            temperature=0.3,
        )
        final_text = st.write_stream(stream)

    # (Optional) Show raw weather JSON for transparency
    with st.expander("Raw weather data"):
        if tool_outputs:
            try:
                st.json(json.loads(tool_outputs[0]["content"]))
            except Exception:
                st.write(tool_outputs[0]["content"])
        else:
            st.write("Used fallback fetch above.")

