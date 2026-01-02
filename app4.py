import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(page_title="Indian Stock Dashboard", layout="wide")

# --- Helper Functions ---
def get_stock_data(ticker, period="5y"):
    # Append .NS for NSE stocks if not present (common for Indian stocks)
    if not (ticker.endswith(".NS") or ticker.endswith(".BO")):
        ticker = f"{ticker}.NS"
        
    data = yf.download(ticker, period=period, interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data, ticker

def get_fundamental_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "trailingEPS": info.get("trailingEps"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "sector": info.get("sector", "N/A"),
            "beta": info.get("beta", "N/A")
        }
    except:
        return {}

def calculate_technical_indicators(df):
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['SMA20'] - (df['STD20'] * 2)
    
    return df

def get_trading_signals(df):
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(df)):
        if df['MACD'].iloc[i] > df['Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1]:
            buy_signals.append(df['Close'].iloc[i])
            sell_signals.append(None)
        elif df['MACD'].iloc[i] < df['Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['Signal'].iloc[i-1]:
            sell_signals.append(df['Close'].iloc[i])
            buy_signals.append(None)
        else:
            buy_signals.append(None)
            sell_signals.append(None)
            
    buy_signals.insert(0, None)
    sell_signals.insert(0, None)
    
    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    return df

def calculate_risk_metrics(df):
    daily_returns = df['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    roll_max = df['Close'].cummax()
    drawdown = df['Close'] / roll_max - 1.0
    max_drawdown = drawdown.min() * 100
    
    return volatility, max_drawdown

def calculate_performance(df):
    current_price = df['Close'].iloc[-1]
    def get_change(days):
        if len(df) > days:
            past_price = df['Close'].iloc[-days-1]
            return ((current_price - past_price) / past_price) * 100
        return 0.0

    return {
        "1 Week": get_change(5),
        "1 Month": get_change(21),
        "6 Months": get_change(126),
        "1 Year": get_change(252),
        "5 Years": get_change(252*5)
    }

def calculate_support_resistance(df, current_price, tolerance_pct=0.15):
    supports = []
    resistances = []
    
    for i in range(2, len(df) - 2):
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i-2] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and df['Low'].iloc[i] < df['Low'].iloc[i+2]):
            supports.append(df['Low'].iloc[i])
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i-2] and 
            df['High'].iloc[i] > df['High'].iloc[i+1] and df['High'].iloc[i] > df['High'].iloc[i+2]):
            resistances.append(df['High'].iloc[i])

    lb = current_price * (1 - tolerance_pct)
    ub = current_price * (1 + tolerance_pct)
    relevant_sup = [s for s in supports if s > lb and s < current_price]
    relevant_res = [r for r in resistances if r < ub and r > current_price]

    def consolidate(levels):
        if not levels: return []
        levels.sort()
        merged = []
        curr_grp = [levels[0]]
        for i in range(1, len(levels)):
            if levels[i] <= curr_grp[-1] * 1.02:
                curr_grp.append(levels[i])
            else:
                merged.append(sum(curr_grp)/len(curr_grp))
                curr_grp = [levels[i]]
        merged.append(sum(curr_grp)/len(curr_grp))
        return merged

    return consolidate(relevant_sup), consolidate(relevant_res)

# --- Dashboard Layout ---
st.title("📈 Stock Dashboard (NSE/BSE)")

# Sidebar
st.sidebar.header("Configuration")
# Defaulting to common Indian stocks (Reliance, TCS, HDFC Bank, Infosys)
tickers_input = st.sidebar.text_input("Tickers", "BHARTIARTL,M&M,VBL,GESHIP,ICICIBANK,CUMMINSIND,TATAPOWER,LICHSGFIN,KOTAKBANK,INDUSTOWER,EXIDEIND,MAHSEAMLES,UJJIVANSFB,JSWINFRA,INDUSINDBK,LTFOODS,RATEGAIN,KFINTECH,PGEL,GHCL,REDINGTON,TATASTEEL,IEX,ZYDUSLIFE,RKFORGE,TATACOMM,CASTROLIND,SONATSOFTW,BIOCON,BLUESTARCO,GENUSPOWER,HYUNDAI,LGEINDIA,NSDL.BO")
ticker_list = [x.strip().upper() for x in tickers_input.split(",")]

st.sidebar.markdown("---")
st.sidebar.subheader("P/E Thresholds")
buy_pe = st.sidebar.number_input("Target Buy P/E", value=20)
sell_pe = st.sidebar.number_input("Target Sell P/E", value=45)

# Container for Export Data
export_data = []

for raw_ticker in ticker_list:
    st.markdown("---")
    
    try:
        # 1. Fetch & Calculate
        # Pass raw_ticker; function returns data and the actual symbol used (e.g. RELIANCE -> RELIANCE.NS)
        df, symbol = get_stock_data(raw_ticker)
        
        if df.empty:
            st.warning(f"No data for {raw_ticker}. Try adding .NS or .BO if needed.")
            continue

        st.header(f"Analysis: {symbol}")
        
        fund = get_fundamental_info(symbol)
        df = calculate_technical_indicators(df)
        df = get_trading_signals(df)
        perf = calculate_performance(df)
        volatility, max_dd = calculate_risk_metrics(df)
        
        curr_price = df['Close'].iloc[-1]
        sup, res = calculate_support_resistance(df, curr_price)
        
        # Historical PE
        eps = fund.get('trailingEPS')
        if eps and eps > 0:
            df['Hist_PE'] = df['Close'] / eps
        else:
            df['Hist_PE'] = 0

        # --- DATA COLLECTION FOR EXPORT ---
        export_data.append({
            "Ticker": symbol,
            "Current Price": round(curr_price, 2),
            "1 Month Return (%)": round(perf['1 Month'], 2),
            "1 Year Return (%)": round(perf['1 Year'], 2),
            "Volatility (%)": round(volatility, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "P/E Ratio": round(fund.get('trailingPE'), 2) if fund.get('trailingPE') else None,
            "Forward P/E": round(fund.get('forwardPE'), 2) if fund.get('forwardPE') else None,
            "Support Levels": " | ".join([str(round(x,1)) for x in sup]),
            "Resistance Levels": " | ".join([str(round(x,1)) for x in res])
        })
        # ----------------------------------

        # 2. Metrics Display (Updated with ₹)
        st.subheader("Performance & Risk Profile")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.metric("Current Price", f"₹{curr_price:,.2f}")
        c2.metric("1 Year Return", f"{perf['1 Year']:.1f}%", delta_color="normal")
        c3.metric("Volatility", f"{volatility:.1f}%", help="Annualized Standard Deviation")
        c4.metric("Max Drawdown", f"{max_dd:.1f}%", delta_color="inverse")
        
        # P/E Color Logic
        pe_val = fund.get('trailingPE')
        pe_str = f"{pe_val:.1f}" if pe_val else "N/A"
        pe_label = "HOLD"
        if pe_val:
            if pe_val < buy_pe: pe_label = "BUY (Cheap)"
            elif pe_val > sell_pe: pe_label = "SELL (Exp)"
        
        c5.metric("P/E Ratio", pe_str, pe_label)
        c6.metric("Forward P/E", f"{fund.get('forwardPE'):.1f}" if fund.get('forwardPE') else "N/A")

        # 3. Visualization
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.06, row_heights=[0.5, 0.2, 0.3],
                            subplot_titles=(f'Price (₹) & Levels', 'MACD Momentum', 'Historical P/E Ratio'))

        # Row 1: Price + BB + Signals
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', showlegend=False), row=1, col=1)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Signal'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=12)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Sell_Signal'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=12)), row=1, col=1)

        for s in sup: fig.add_hline(y=s, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)
        for r in res: fig.add_hline(y=r, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)

        # Row 2: MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], marker_color=['green' if v >= 0 else 'red' for v in df['Histogram']], name='Hist'), row=2, col=1)

        # Row 3: PE
        if eps and eps > 0:
            fig.add_trace(go.Scatter(x=df.index, y=df['Hist_PE'], name='P/E', line=dict(color='purple'), fill='tozeroy'), row=3, col=1)
            fig.add_hline(y=buy_pe, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=sell_pe, line_dash="dash", line_color="red", row=3, col=1)
        else:
            fig.add_annotation(text="No EPS Data", showarrow=False, row=3, col=1)

        fig.update_layout(height=850, xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing {raw_ticker}: {e}")

# --- Export Button Logic ---
if export_data:
    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Report (CSV)",
        data=csv,
        file_name="stock_analysis_report.csv",
        mime="text/csv",
    )