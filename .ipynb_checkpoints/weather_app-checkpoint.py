from googlesearch import search
from typing import Literal
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from api_token import Falcon_API
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.prompts import BasePromptTemplate
from langchain.chains import RouterChain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

api = Falcon_API()
api_key = api.get_Key()

llm = HuggingFaceEndpoint(
    name="WEB_PILOT",
    huggingfacehub_api_token=api_key,
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task="text-generation",
    max_new_tokens=500,
    temperature=0.1
)

# Configuration and setup
st.set_page_config(page_title="Weather AI Assistant", page_icon="🌦️")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .header {
            color: #1f77b4;
            text-align: center;
            padding: 10px;
        }
        .metric-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .highlight {
            color: #ff4b4b;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper function to scrape text from a URL
def scrape_text(url, num_words):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = " ".join([p.text for p in soup.find_all("p")])
        return " ".join(text.split()[:num_words])
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return None

# Function to fetch weather data
def get_weather_data(query):
    try:
        search_results = list(search(query, num_results=5))
        for url in search_results:
            scraped_data = scrape_text(url, num_words=200)
            if scraped_data:
                prompt = """You are an expert weather assistant. Analyze the following weather information and provide it in structured JSON format with details such as location, temperature, wind, precipitation, humidity, UV index, and visibility:
                {data}
                """

                chain = (
                    PromptTemplate.from_template(prompt)
                    | llm  # Ensure your LLM is properly initialized
                    | JsonOutputParser()
                )

                parsed_data = chain.invoke({"data": scraped_data})
                if parsed_data:
                    return parsed_data

        return None
    except Exception as e:
        st.error(f"An error occurred while fetching weather data: {e}")
        return None

# Main Streamlit App
st.title("🌦️ Smart Weather Assistant")
st.markdown("---")

# User Input
query = st.text_input(
    "Enter your weather query:", placeholder="e.g., What's the weather in Tokyo tomorrow?"
)

if st.button("Get Weather", type="primary"):
    if not query:
        st.warning("Please enter a weather query!")
    else:
        with st.spinner("🌤️ Fetching weather data..."):
            weather_data = get_weather_data(query)

            if not weather_data:
                st.error("Could not retrieve weather data. Please try again.")
            else:
                # Display Weather Data
                st.markdown(
                    f"<h2 class='header'>Weather Report for {weather_data.get('location', 'N/A')}</h2>",
                    unsafe_allow_html=True
                )

                # Layout with Columns
                col1, col2, col3 = st.columns(3)

                # Temperature Card
                with col1:
                    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                    st.subheader("🌡️ Temperature")
                    temp = weather_data.get('temperature', {})
                    st.metric("Current", f"{temp.get('current', 'N/A')}°C")
                    st.write(f"Feels like: {temp.get('feels_like', 'N/A')}°C")
                    st.write(f"Min/Max: {temp.get('min', 'N/A')}°C / {temp.get('max', 'N/A')}°C")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Wind Card
                with col2:
                    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                    st.subheader("🌬️ Wind")
                    wind = weather_data.get('wind', {})
                    st.metric("Speed", f"{wind.get('speed', 'N/A')} km/h")
                    st.write(f"Direction: {wind.get('direction', 'N/A')}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Precipitation Card
                with col3:
                    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                    st.subheader("🌧️ Precipitation")
                    precip = weather_data.get('precipitation', {})
                    st.metric("Chance", f"{precip.get('chance', 'N/A')}%")
                    st.write(f"Type: {precip.get('type', 'N/A')}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Additional Information
                st.markdown("---")
                st.subheader("Additional Details")
                st.write(f"**Humidity:** {weather_data.get('humidity', 'N/A')}%")
                st.write(f"**UV Index:** {weather_data.get('uv_index', 'N/A')}")
                st.write(f"**Visibility:** {weather_data.get('visibility', 'N/A')} km")
