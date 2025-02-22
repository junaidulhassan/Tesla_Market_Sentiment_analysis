import streamlit as st
import requests
import json
import re
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import tensorflow as tf
from datetime import datetime
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
import os
import warnings

# Configuration
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Tesla Stock Analysis", layout='wide')
MODEL_PATH = "Tesla_model.h5"
TICKER_SYMBOL = "TSLA"
FORECAST_DAYS = 7

# Initialize API keys (assuming proper implementation in api_token.py)
from api_token import LargeLanguageModel
api = LargeLanguageModel()
# os.environ['OPENAI_API_KEY'] = api.get_gpt_key()

# # Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini")

api = LargeLanguageModel()
api_key = api.get_Key()

llm = HuggingFaceEndpoint(
    name="WEB_PILOT",
    huggingfacehub_api_token=api_key,
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.3',
    task="text-generation",
    max_new_tokens=1000,
    temperature=0.1
)

def load_stock_model():
    """Load and compile the stock prediction model"""
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss=tf.keras.losses.MeanSquaredError())
    return model

def fetch_stock_data(days=20):
    """Fetch historical stock data"""
    ticker = yf.Ticker(TICKER_SYMBOL)
    hist = ticker.history(period=f"{days}d")["Close"]
    return hist

def generate_predictions(model, historical_data, forecast_days=7):
    """Generate stock price predictions using the loaded model"""
    current_prices = list(historical_data)
    for _ in range(forecast_days):
        input_data = np.array(current_prices[-7:]).reshape(1, 7, 1)
        prediction = model.predict(input_data, verbose=0)
        current_prices.append(prediction[0][0])
    return current_prices

def create_stock_chart(historical_data, predictions, hist_dates):
    """Create interactive Plotly chart with historical and predicted data."""
    
    # Generate prediction dates
    last_date = hist_dates[-1]
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    
    # Create figure
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=historical_data,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add transition segment to connect last real point to first predicted point
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], pred_dates[0]],
        y=[historical_data[-1], predictions[0]],
        mode='lines',
        line=dict(color='#ff7f0e', width=2),
        showlegend=False  
    ))
    
    # Predicted data
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Data',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{TICKER_SYMBOL} Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    
    return fig

def scrape_website(url):
    """Scrape website content with error handling"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        scraped_text = soup.get_text(separator=' ', strip=True)
        return scraped_text[4000:-2000]
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return None

def generate_insights(stock_data):
    """Generate market insights using LLM"""
    prompt_template = PromptTemplate.from_template(
        """
        You are an AI Stock Assistant specializing in Tesla market insights. Your task is to analyze the latest Tesla stock data and provide insights on market trends, news, and potential stock movements.
        Based on the given data, consider these key factors:
        - Today's stock price trends.
        - Market sentiment and news impact.
        - Technical analysis indicators.
        - Economic and industry factors.
        - Predictions on price movement (up/down).
        
        <analysis>
        {data}
        </analysis>

        Provide a structured JSON response with the following keys:
        {{
            "trend_analysis": "Summary of stock price trends",
            "market_sentiment": "Impact of news and general sentiment",
            "technical_indicators": "Technical analysis insights",
            "economic_factors": "Economic and industry-related factors",
            "prediction": "Predicted stock movement: up or down"
        }}

        Return just valid JSON.
        """
    )
    
    chain = LLMChain(prompt=prompt_template, llm=llm)
    try:
        response = chain.invoke({"data": stock_data})
        return parse_json_response(response['text'])
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

def parse_json_response(text):
    """Parse JSON response from LLM output"""
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except json.JSONDecodeError:
        st.error("Failed to parse JSON response")
        return None

def display_insights(insights):
    """Display formatted insights in Streamlit"""
    if not insights:
        return
    
    with st.expander("📊 Trend Analysis"):
        st.write(insights.get("trend_analysis", "N/A"))
    
    with st.expander("📈 Technical Indicators"):
        st.write(insights.get("technical_indicators", "N/A"))
    
    with st.expander("💹 Market Sentiment"):
        st.write(insights.get("market_sentiment", "N/A"))
    
    with st.expander("🌍 Economic Factors"):
        st.write(insights.get("economic_factors", "N/A"))
    
    with st.expander("🔮 Price Prediction"):
        st.write(insights.get("prediction", "N/A"))

def style_metric(value, label, delta=None):
    """Styled metric component with shadow effect"""
    return f"""
    <div style="
        padding: 20px;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;">
        <div style="font-size: 14px; color: #666;">{label}</div>
        <div style="font-size: 24px; font-weight: bold; color: #2e86de;">{value}</div>
    </div>
    """

# Main application
def main():
    st.title("📈 Tesla Stock Analysis & Prediction")
    st.write(f"#### {datetime.today().strftime('%A, %B %d, %Y')}")
    
    # Fetch real-time data
    ticker = yf.Ticker(TICKER_SYMBOL)
    today_data = ticker.history(period="1d").iloc[0]
    
    # Display metrics in columns
    cols = st.columns(5)
    metrics = [
        (f"${today_data['Close']:.2f}", "Closing Price"),
        (f"${today_data['High']:.2f}", "Daily High"),
        (f"${today_data['Low']:.2f}", "Daily Low"),
        (f"${today_data['Open']:.2f}", "Opening Price"),
        (f"{today_data['Volume']:,.0f}", "Volume")
    ]
    
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(style_metric(value, label), unsafe_allow_html=True)


    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        analysis_url = st.text_input(
            "News Source URL:", 
            "https://finance.yahoo.com/quote/TSLA/analysis/"
        )
        analyze_btn = st.button("Analyze Stock")
    
    if analyze_btn:
        with st.spinner("Analyzing market data..."):
            # Load model and data
            model = load_stock_model()
            hist_data = fetch_stock_data()
            hist_dates = hist_data.index
            
            # Generate predictions
            full_predictions = generate_predictions(model, hist_data.values)
            predictions = full_predictions[-FORECAST_DAYS:]
            
            # Create and display chart
            fig = create_stock_chart(
                hist_data.values, 
                predictions, 
                hist_dates
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction table
            pred_dates = pd.date_range(
                start=hist_dates[-1], 
                periods=FORECAST_DAYS+1
            )[1:]
            
            st.subheader("Next 7 Days Predictions")
            pred_df = pd.DataFrame({
                "Date": pred_dates.strftime("%Y-%m-%d"),
                "Predicted Price": [f"${x:.2f}" for x in predictions]
            })
            st.dataframe(pred_df, hide_index=True)
            
            # Generate and display insights
            scraped_data = scrape_website(analysis_url)
            if scraped_data:
                insights = generate_insights(scraped_data)
                if insights:
                    st.subheader("Market Insights")
                    display_insights(insights)
                else:
                    st.warning("No insights generated")
            else:
                st.error("Failed to retrieve analysis data")

if __name__ == "__main__":
    main()