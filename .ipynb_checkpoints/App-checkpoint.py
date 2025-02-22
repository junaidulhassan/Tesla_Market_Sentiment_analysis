import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from api_token import LargeLanguageModel
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import warnings as wn
wn.filterwarnings('ignore')

# Initialize API keys
api = LargeLanguageModel()
os.environ['OPENAI_API_KEY'] = api.get_gpt_key()
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Streamlit UI
st.set_page_config(page_title="Tesla Stock Analysis", layout="wide")
st.title("📈 Tesla Stock Market Insights")

# Display today's date
today_date = datetime.today().strftime('%Y-%m-%d')
st.write(f"### 📅 Today: {today_date}")

# Sidebar for input
with st.sidebar:
    st.header("Settings")
    url = st.text_input("Enter Stock URL:", "https://www.cnbc.com/quotes/TSLA?tab=profile")
    analyze_button = st.button("Analyze Stock Data")

# Function to scrape webpage
def scrape_page(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    return None

# Prompt Template
prompt_template = PromptTemplate.from_template(
    """
    You are an AI Stock Assistant specializing in Tesla market insights. Your task is to analyze the latest Tesla stock data and provide insights on market trends, news, and potential stock movements.
    
    Based on the given data, extract and analyze the following key factors:
    1. Important stock insights and trends.
    2. Market sentiment and latest news impact.
    3. Technical indicators (like RSI, MACD, moving averages if available).
    4. Predict tomorrow's stock movement (up/down) based on trends.
    
    <analysis>
    {data}
    </analysis>
    
    Provide structured insights including key metrics, trends, and a brief prediction.
    
    Insights:
    """
)

# Function to generate insights using AI
def generate_insights(data):
    chain = LLMChain(prompt=prompt_template, llm=llm)
    analysis = chain.invoke({"data": data})
    return analysis['text']

# Function to plot stock insights
def plot_stock_trends():
    dates = np.array(["Day -3", "Day -2", "Yesterday", "Today"])
    stock_prices = np.random.uniform(200, 250, size=4)  # Randomized example data
    
    fig, ax = plt.subplots()
    ax.plot(dates, stock_prices, marker='o', linestyle='-', color='b', label='Stock Price')
    ax.set_title("Tesla Stock Trend Over Last 4 Days")
    ax.set_ylabel("Stock Price (USD)")
    ax.legend()
    
    st.pyplot(fig)

# Run analysis when button is clicked
if analyze_button:
    st.subheader("🔄 Scraping Data...")
    scraped_text = scrape_page(url)
    
    if scraped_text:
        st.success("✅ Data Scraped Successfully!")
        st.subheader("📊 Key Stock Insights")
        
        insights = generate_insights(scraped_text)
        st.write(insights)
        
        # Display stock trends graph
        st.subheader("📉 Stock Price Trends")
        plot_stock_trends()
        
        # Show market sentiment & insights
        st.subheader("📰 Market Sentiment & Insights")
        st.write("🔹 Market sentiment analysis based on news and trends.")
        
        # Predict next day's stock movement
        st.subheader("📈 Next Day Prediction")
        st.write("Based on current trends, the stock is likely to move [Up/Down].")
    else:
        st.error("❌ Failed to scrape data. Check the URL and try again.")
