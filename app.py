import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from prophet import Prophet
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="Agricultural Market Analysis",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
    }
    .subheader {
        font-size: 1.5rem;
        color: #43a047;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f1f8e9;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Agricultural Market Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("This dashboard provides insights into agricultural commodity prices, trends, and forecasts.")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Price Analysis", "Forecasting", "Weather Impact"])

# Sample agricultural commodities
commodities = {
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Coffee": "KC=F",
    "Cotton": "CT=F",
    "Sugar": "SB=F",
    "Rice": "ZR=F"
}

# Function to fetch data
@st.cache_data(ttl=3600)
def fetch_commodity_data(symbol, period="1y"):
    try:
        data = yf.download(symbol, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to create forecast
def create_forecast(data, days=30):
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    return forecast, model

# Market Overview Tab
with tab1:
    st.markdown('<p class="subheader">Current Market Prices</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_commodity = st.selectbox("Select a commodity:", list(commodities.keys()))
    
    with col2:
        period = st.selectbox("Select time period:", ["1m", "3m", "6m", "1y", "2y", "5y"])
    
    # Fetch data
    symbol = commodities[selected_commodity]
    data = fetch_commodity_data(symbol, period)
    
    if not data.empty:
        # Display current price and daily change
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_percent = (price_change / prev_price) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(f"{selected_commodity} Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_percent:.2f}%)")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("52-Week Range", f"${data['Low'].min():.2f} - ${data['High'].max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Price chart
        st.markdown('<p class="subheader">Price Chart</p>', unsafe_allow_html=True)
        fig = px.line(
            data,
            x=data.index,
            y="Close",
            title=f"{selected_commodity} Price History",
            labels={"Close": "Price (USD)", "index": "Date"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market summary
        st.markdown('<p class="subheader">Market Summary</p>', unsafe_allow_html=True)
        
        # Calculate some basic statistics
        avg_price = data['Close'].mean()
        max_price = data['Close'].max()
        min_price = data['Close'].min()
        volatility = data['Close'].pct_change().std() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Key Statistics")
            st.markdown(f"**Average Price:** ${avg_price:.2f}")
            st.markdown(f"**Maximum Price:** ${max_price:.2f}")
            st.markdown(f"**Minimum Price:** ${min_price:.2f}")
            st.markdown(f"**Volatility:** {volatility:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Price Change Over Selected Period")
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            total_change = end_price - start_price
            total_percent = (total_change / start_price) * 100
            
            if total_change >= 0:
                st.markdown(f"**Price Change:** 📈 +${total_change:.2f} (+{total_percent:.2f}%)")
                st.markdown(f"**Trend:** Upward trend over the selected period")
            else:
                st.markdown(f"**Price Change:** 📉 ${total_change:.2f} ({total_percent:.2f}%)")
                st.markdown(f"**Trend:** Downward trend over the selected period")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Failed to fetch data. Please try again later.")

# Price Analysis Tab
with tab2:
    st.markdown('<p class="subheader">Comparative Price Analysis</p>', unsafe_allow_html=True)
    
    # Select multiple commodities to compare
    selected_commodities = st.multiselect(
        "Select commodities to compare:",
        list(commodities.keys()),
        default=[list(commodities.keys())[0], list(commodities.keys())[1]]
    )
    
    compare_period = st.selectbox("Select comparison period:", ["1m", "3m", "6m", "1y"], key="compare_period")
    
    if selected_commodities:
        # Create comparison dataframe
        comparison_df = pd.DataFrame()
        
        for commodity in selected_commodities:
            symbol = commodities[commodity]
            data = fetch_commodity_data(symbol, compare_period)
            if not data.empty:
                comparison_df[commodity] = data['Close']
        
        if not comparison_df.empty:
            # Normalize prices for better comparison
            normalized_df = comparison_df.div(comparison_df.iloc[0]) * 100
            
            # Plot comparative chart
            fig_compare = px.line(
                normalized_df,
                x=normalized_df.index,
                y=normalized_df.columns,
                title="Normalized Price Comparison (Base 100)",
                labels={"value": "Normalized Price", "index": "Date", "variable": "Commodity"}
            )
            fig_compare.update_layout(height=500)
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Correlation matrix
            st.markdown('<p class="subheader">Correlation Matrix</p>', unsafe_allow_html=True)
            corr_matrix = comparison_df.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Price Correlation Between Commodities"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Price volatility comparison
            st.markdown('<p class="subheader">Price Volatility Comparison</p>', unsafe_allow_html=True)
            
            volatility_data = {}
            for commodity in selected_commodities:
                volatility_data[commodity] = comparison_df[commodity].pct_change().std() * 100
                
            volatility_df = pd.DataFrame({
                'Commodity': list(volatility_data.keys()),
                'Volatility (%)': list(volatility_data.values())
            })
            
            fig_vol = px.bar(
                volatility_df,
                x='Commodity',
                y='Volatility (%)',
                title="Price Volatility Comparison",
                color='Volatility (%)',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.error("Failed to fetch comparison data. Please try again.")
    else:
        st.info("Please select at least one commodity for analysis.")

# Forecasting Tab
with tab3:
    st.markdown('<p class="subheader">Price Forecasting</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_commodity = st.selectbox("Select a commodity for forecasting:", list(commodities.keys()), key="forecast_commodity")
    
    with col2:
        forecast_days = st.slider("Forecast days:", min_value=7, max_value=90, value=30, step=7)
    
    # Fetch data for forecasting
    forecast_symbol = commodities[forecast_commodity]
    forecast_data = fetch_commodity_data(forecast_symbol, "2y")  # Use more historical data for better forecasting
    
    if not forecast_data.empty:
        with st.spinner(f"Generating forecast for {forecast_commodity}..."):
            try:
                # Create forecast
                forecast, model = create_forecast(forecast_data, forecast_days)
                
                # Plot forecast
                fig_forecast = go.Figure()
                
                # Add historical data
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                # Add confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(width=0),
                    name='Confidence Interval'
                ))
                
                fig_forecast.update_layout(
                    title=f"{forecast_commodity} Price Forecast for Next {forecast_days} Days",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Show forecast components
                st.markdown('<p class="subheader">Forecast Components</p>', unsafe_allow_html=True)
                
                # Create components figure
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
                
                # Display forecast metrics
                st.markdown('<p class="subheader">Forecast Summary</p>', unsafe_allow_html=True)
                
                last_price = forecast_data['Close'].iloc[-1]
                forecast_price = forecast['yhat'].iloc[-1]
                price_change = forecast_price - last_price
                price_change_percent = (price_change / last_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Current Price", f"${last_price:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric(f"Forecasted Price ({forecast_days} days)", f"${forecast_price:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Expected Change", f"{price_change:.2f} ({price_change_percent:.2f}%)")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.info("Forecasting requires the Prophet package. Please make sure it's installed.")
    else:
        st.error("Failed to fetch data for forecasting. Please try again later.")

# Weather Impact Tab
with tab4:
    st.markdown('<p class="subheader">Weather Impact Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This section analyzes the correlation between weather patterns and agricultural commodity prices.
    Weather data can be a significant factor affecting crop yields and prices.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        weather_commodity = st.selectbox("Select commodity:", list(commodities.keys()), key="weather_commodity")
    
    with col2:
        weather_metric = st.selectbox("Select weather metric:", ["Temperature", "Precipitation", "Drought Index"])
    
    # For demonstration, we'll use synthetic weather data
    # In a real app, you would fetch this from a weather API
    
    # Generate synthetic weather data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    
    if weather_metric == "Temperature":
        # Simulate seasonal temperature pattern
        base_temp = 20  # base temperature in celsius
        amplitude = 10  # seasonal variation
        noise = np.random.normal(0, 2, size=len(dates))  # random variations
        temperatures = base_temp + amplitude * np.sin(np.linspace(0, 2*np.pi, len(dates))) + noise
        
        weather_df = pd.DataFrame({
            'Date': dates,
            'Temperature': temperatures
        })
        
        # Plot temperature data
        fig_weather = px.line(
            weather_df,
            x='Date',
            y='Temperature',
            title="Average Daily Temperature (°C)",
            labels={"Temperature": "Temperature (°C)", "Date": "Date"}
        )
        st.plotly_chart(fig_weather, use_container_width=True)
        
    elif weather_metric == "Precipitation":
        # Simulate seasonal precipitation pattern
        base_precip = 5  # base precipitation in mm
        amplitude = 3  # seasonal variation
        noise = np.random.exponential(2, size=len(dates))  # random variations
        precipitation = base_precip + amplitude * np.sin(np.linspace(0, 2*np.pi, len(dates))) + noise
        precipitation = np.maximum(precipitation, 0)  # precipitation can't be negative
        
        weather_df = pd.DataFrame({
            'Date': dates,
            'Precipitation': precipitation
        })
        
        # Plot precipitation data
        fig_weather = px.bar(
            weather_df,
            x='Date',
            y='Precipitation',
            title="Daily Precipitation (mm)",
            labels={"Precipitation": "Precipitation (mm)", "Date": "Date"}
        )
        st.plotly_chart(fig_weather, use_container_width=True)
        
    else:  # Drought Index
        # Simulate drought index
        base_index = 0  # neutral
        trend = np.linspace(0, 2, len(dates))  # gradual trend
        noise = np.random.normal(0, 0.5, size=len(dates))  # random variations
        drought_index = base_index + 2 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + trend * 0.5 + noise
        
        weather_df = pd.DataFrame({
            'Date': dates,
            'Drought_Index': drought_index
        })
        
        # Plot drought index
        fig_weather = px.line(
            weather_df, 
            x='Date',
            y='Drought_Index',
            title="Palmer Drought Severity Index",
            labels={"Drought_Index": "PDSI", "Date": "Date"}
        )
        
        # Add reference lines
        fig_weather.add_hline(y=4, line_dash="dash", line_color="green", annotation_text="Extremely Wet")
        fig_weather.add_hline(y=2, line_dash="dash", line_color="lightgreen", annotation_text="Moderately Wet")
        fig_weather.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Near Normal")
        fig_weather.add_hline(y=-2, line_dash="dash", line_color="orange", annotation_text="Moderate Drought")
        fig_weather.add_hline(y=-4, line_dash="dash", line_color="red", annotation_text="Extreme Drought")
        
        st.plotly_chart(fig_weather, use_container_width=True)
    
    # Fetch price data for correlation analysis
    weather_symbol = commodities[weather_commodity]
    price_data = fetch_commodity_data(weather_symbol, "1y")
    
    if not price_data.empty:
        # Resample price data to daily frequency for joining with weather data
        daily_price = price_data['Close'].resample('D').last().fillna(method='ffill')
        
        # Create a common date range
        common_dates = sorted(set(daily_price.index) & set(weather_df['Date']))
        
        if common_dates:
            correlation_df = pd.DataFrame({
                'Date': common_dates,
                'Price': [daily_price.loc[date] for date in common_dates],
                'Weather_Metric': [weather_df.loc[weather_df['Date'] == date].iloc[0][weather_metric if weather_metric != "Drought Index" else "Drought_Index"] for date in common_dates]
            })
            
            # Calculate correlation
            corr = correlation_df['Price'].corr(correlation_df['Weather_Metric'])
            
            st.markdown(f"### Correlation Analysis: {weather_commodity} Prices vs {weather_metric}")
            st.markdown(f"**Correlation Coefficient:** {corr:.3f}")
            
            if abs(corr) < 0.3:
                st.markdown("**Interpretation:** Weak correlation. This suggests that the selected weather metric has limited impact on this commodity's price.")
            elif abs(corr) < 0.7:
                st.markdown("**Interpretation:** Moderate correlation. This suggests that the selected weather metric has some impact on this commodity's price.")
            else:
                st.markdown("**Interpretation:** Strong correlation. This suggests that the selected weather metric has a significant impact on this commodity's price.")
            
            # Scatter plot
            fig_scatter = px.scatter(
                correlation_df,
                x='Weather_Metric',
                y='Price',
                title=f"{weather_commodity} Price vs {weather_metric}",
                trendline="ols",
                labels={"Weather_Metric": weather_metric if weather_metric != "Drought Index" else "Drought Index", "Price": f"{weather_commodity} Price (USD)"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            st.error("No common dates found between price and weather data.")
    else:
        st.error("Failed to fetch price data for correlation analysis.")

# Footer
st.markdown("---")
st.markdown("### About This Dashboard")
st.markdown("""
This Agricultural Market Analysis Dashboard provides real-time and historical data on agricultural commodity prices, 
along with advanced analytics including price forecasting and weather impact analysis.

**Data Sources:**
- Commodity price data: Yahoo Finance
- Weather data: Synthetic (for demonstration)

**Features:**
- Real-time market prices and trends
- Comparative analysis of multiple commodities
- Price forecasting using Prophet algorithm
- Weather impact analysis

**Note:** This is a demonstration application. For production use, consider integrating with specialized agricultural data APIs.
""")
