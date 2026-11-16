import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ==========================================
# CHAMPION GOLD BOT
# Educational XAUUSD SMC analysis dashboard
# ==========================================

st.set_page_config(
    page_title="CHAMPION GOLD BOT",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 CHAMPION GOLD BOT")
st.caption("Gold market analysis | Smart Money Concepts")
st.info(
    "Educational analysis only. Signals are setup candidates, "
    "not financial advice or guaranteed trade calls."
)

# Refresh the dashboard every 60 seconds
st_autorefresh(interval=60 * 1000, key="gold_refresh")


# ==========================================
# DATA FUNCTIONS
# ==========================================

@st.cache_data(ttl=60)
def get_yahoo_data():
    """Try Yahoo Finance. This may not provide XAUUSD 15m data."""
    ticker = yf.Ticker("XAUUSD=X")
    data = ticker.history(
        period="30d",
        interval="15m",
        auto_adjust=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()
    data.index = pd.to_datetime(data.index)
    data = data.dropna()
    data = data[~data.index.duplicated(keep="last")]
    return data.sort_index()


def load_csv(uploaded_file):
    """Load user-provided XAUUSD OHLC candles."""
    data = pd.read_csv(uploaded_file)

    data.columns = [str(c).strip().lower() for c in data.columns]

    time_column = None
    for name in ["datetime", "timestamp", "date", "time"]:
        if name in data.columns:
            time_column = name
            break

    if time_column is None:
        raise ValueError(
            "CSV needs a Date, Datetime, Timestamp or Time column."
        )

    required = ["open", "high", "low", "close"]
    if not all(col in data.columns for col in required):
        raise ValueError(
            "CSV needs Open, High, Low and Close columns."
        )

    data[time_column] = pd.to_datetime(
        data[time_column], errors="coerce"
    )

    data = data.dropna(subset=[time_column])
    data = data.set_index(time_column)

    data = data[["open", "high", "low", "close"]].copy()
    data.columns = ["Open", "High", "Low", "Close"]

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    if len(data) < 100:
        raise ValueError(
            "Please upload at least 100 candles for analysis."
        )

    return data


# ==========================================
# MARKET STRUCTURE
# ==========================================

def find_confirmed_swings(data, left=2, right=2):
    """
    A swing is only confirmed after the candles to its right
    have closed. This reduces the use of unconfirmed pivots.
    """
    highs = []
    lows = []

    high_values = data["High"].to_numpy()
    low_values = data["Low"].to_numpy()

    for i in range(left + right, len(data)):
        j = i - right

        high_window = high_values[j-left:j+right+1]
        low_window = low_values[j-left:j+right+1]

        if high_values[j] == max(high_window):
            highs.append((j, high_values[j]))

        if low_values[j] == min(low_window):
            lows.append((j, low_values[j]))

    return highs, lows


def get_structure(data):
    highs, lows = find_confirmed_swings(data)

    if len(highs) < 2 or len(lows) < 2:
        return "NOT ENOUGH STRUCTURE", None, None

    previous_high = highs[-2][1]
    latest_high = highs[-1][1]

    previous_low = lows[-2][1]
    latest_low = lows[-1][1]

    if latest_high > previous_high and latest_low > previous_low:
        label = "HH + HL — BULLISH STRUCTURE"
    elif latest_high < previous_high and latest_low < previous_low:
        label = "LH + LL — BEARISH STRUCTURE"
    else:
        label = "MIXED / SIDEWAYS STRUCTURE"

    return label, highs[-1], lows[-1]


# ==========================================
# HIGHER TIMEFRAME BIAS
# ==========================================

def get_htf_bias(data):
    hourly = data.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    # Exclude the latest hourly bucket because it may be incomplete
    if len(hourly) > 1:
        hourly = hourly.iloc[:-1]

    if len(hourly) < 25:
        return "NOT ENOUGH HOURLY DATA"

    ema20 = hourly["Close"].ewm(span=20, adjust=False).mean()
    ema50 = hourly["Close"].ewm(span=50, adjust=False).mean()

    if ema20.iloc[-1] > ema50.iloc[-1]:
        return "BULLISH"

    if ema20.iloc[-1] < ema50.iloc[-1]:
        return "BEARISH"

    return "NEUTRAL"


# ==========================================
# BOS AND CHOCH
# ==========================================

def get_structure_break(data, swing_high, swing_low):
    last = data.iloc[-1]
    previous = data.iloc[-2]

    if swing_high is None or swing_low is None:
        return "NOT CONFIRMED"

    high_level = swing_high[1]
    low_level = swing_low[1]

    if last["Close"] > high_level and previous["Close"] <= high_level:
        return "BULLISH BOS"

    if last["Close"] < low_level and previous["Close"] >= low_level:
        return "BEARISH BOS"

    return "NO NEW BOS ON LAST CANDLE"


# ==========================================
# LIQUIDITY SWEEP
# ==========================================

def get_liquidity_sweep(data, swing_high, swing_low):
    last = data.iloc[-1]

    if swing_high is None or swing_low is None:
        return "NOT CONFIRMED"

    high_level = swing_high[1]
    low_level = swing_low[1]

    if last["High"] > high_level and last["Close"] < high_level:
        return "BUY-SIDE LIQUIDITY SWEPT"

    if last["Low"] < low_level and last["Close"] > low_level:
        return "SELL-SIDE LIQUIDITY SWEPT"

    return "NO SWEEP ON LAST CANDLE"


# ==========================================
# ORDER BLOCK AND RETEST
# ==========================================

def find_order_block(data, direction, search_bars=20):
    """
    Simplified educational OB logic:
    Finds the last opposing candle before a recent
    structure break, then checks for a later retest.
    """
    if len(data) < 10:
        return None

    highs, lows = find_confirmed_swings(data)

    if not highs or not lows:
        return None

    start = max(5, len(data) - search_bars)
    bos_index = None

    for i in range(start, len(data)):
        close = data["Close"].iloc[i]

        if direction == "BULLISH":
            earlier_highs = [h for h in highs if h[0] < i]
            if earlier_highs and close > earlier_highs[-1][1]:
                bos_index = i

        else:
            earlier_lows = [l for l in lows if l[0] < i]
            if earlier_lows and close < earlier_lows[-1][1]:
                bos_index = i

    if bos_index is None or bos_index < 2:
        return None

    ob_index = None

    for i in range(bos_index - 1, max(0, bos_index - 9), -1):
        candle = data.iloc[i]

        if direction == "BULLISH" and candle["Close"] < candle["Open"]:
            ob_index = i
            break

        if direction == "BEARISH" and candle["Close"] > candle["Open"]:
            ob_index = i
            break

    if ob_index is None:
        return None

    ob = data.iloc[ob_index]
    zone_low = float(ob["Low"])
    zone_high = float(ob["High"])

    # Require price to revisit the zone after the BOS candle
    later = data.iloc[bos_index + 1:]

    if later.empty:
        return None

    latest = data.iloc[-1]
    retested = (
        latest["Low"] <= zone_high
        and latest["High"] >= zone_low
    )

    if retested:
        return {
            "low": zone_low,
            "high": zone_high,
            "retested": True
        }

    return {
        "low": zone_low,
        "high": zone_high,
        "retested": False
    }


# ==========================================
# CANDLE CONFIRMATION
# ==========================================

def candle_confirmation(data, direction):
    if len(data) < 2:
        return False, "NOT ENOUGH CANDLES"

    current = data.iloc[-1]
    previous = data.iloc[-2]

    body = abs(current["Close"] - current["Open"])
    full_range = current["High"] - current["Low"]

    if full_range == 0:
        return False, "NO CLEAR CANDLE CONFIRMATION"

    lower_wick = min(current["Open"], current["Close"]) - current["Low"]
    upper_wick = current["High"] - max(current["Open"], current["Close"])

    if direction == "BULLISH":
        engulfing = (
            current["Close"] > current["Open"]
            and previous["Close"] < previous["Open"]
            and current["Close"] >= previous["Open"]
            and current["Open"] <= previous["Close"]
        )

        hammer = (
            lower_wick >= body * 2
            and upper_wick <= body
            and current["Close"] > current["Open"]
        )

        if engulfing:
            return True, "BULLISH ENGULFING"

        if hammer:
            return True, "HAMMER"

    if direction == "BEARISH":
        engulfing = (
            current["Close"] < current["Open"]
            and previous["Close"] > previous["Open"]
            and current["Open"] >= previous["Close"]
            and current["Close"] <= previous["Open"]
        )

        shooting_star = (
            upper_wick >= body * 2
            and lower_wick <= body
            and current["Close"] < current["Open"]
        )

        if engulfing:
            return True, "BEARISH ENGULFING"

        if shooting_star:
            return True, "SHOOTING STAR"

    return False, "NO CONFIRMATION"


# ==========================================
# FULL ANALYSIS
# ==========================================

def analyse_market(data):
    if len(data) < 100:
        raise ValueError("At least 100 candles are required.")

    bias = get_htf_bias(data)
    structure, swing_high, swing_low = get_structure(data)

    bos = get_structure_break(data, swing_high, swing_low)
    sweep = get_liquidity_sweep(data, swing_high, swing_low)

    direction = None

    if bias == "BULLISH":
        direction = "BULLISH"
    elif bias == "BEARISH":
        direction = "BEARISH"

    ob = None
    ob_pass = False
    candle_pass = False
    candle_name = "NOT CHECKED"

    if direction:
        ob = find_order_block(data, direction)

        if ob:
            ob_pass = ob["retested"]

        candle_pass, candle_name = candle_confirmation(
            data, direction
        )

    structure_pass = (
        direction == "BULLISH"
        and "HH + HL" in structure
    ) or (
        direction == "BEARISH"
        and "LH + LL" in structure
    )

    bos_pass = (
        direction == "BULLISH" and bos == "BULLISH BOS"
    ) or (
        direction == "BEARISH" and bos == "BEARISH BOS"
    )

    sweep_pass = (
        direction == "BULLISH"
        and sweep == "SELL-SIDE LIQUIDITY SWEPT"
    ) or (
        direction == "BEARISH"
        and sweep == "BUY-SIDE LIQUIDITY SWEPT"
    )

    checks = {
        "HTF bias aligned": direction is not None,
        "15m structure aligned": structure_pass,
        "BOS confirmed on latest candle": bos_pass,
        "Liquidity sweep aligned": sweep_pass,
        "Order block retest": ob_pass,
        "Candle confirmation": candle_pass
    }

    all_pass = all(checks.values())

    if all_pass and direction == "BULLISH":
        result = "BUY SETUP CANDIDATE"
    elif all_pass and direction == "BEARISH":
        result = "SELL SETUP CANDIDATE"
    else:
        result = "NO COMPLETE SETUP"

    return {
        "bias": bias,
        "structure": structure,
        "bos": bos,
        "sweep": sweep,
        "ob": ob,
        "ob_pass": ob_pass,
        "candle": candle_name,
        "checks": checks,
        "result": result
    }


# ==========================================
# DATA SOURCE SELECTION
# ==========================================

st.sidebar.header("Market data")

source = st.sidebar.radio(
    "Choose data source",
    ["Try free Yahoo data", "Upload XAUUSD CSV"]
)

data = pd.DataFrame()
source_name = ""

if source == "Try free Yahoo data":
    with st.spinner("Checking for XAUUSD data..."):
        try:
            data = get_yahoo_data()
            if not data.empty:
                source_name = "Yahoo Finance — best effort"
        except Exception:
            data = pd.DataFrame()

    if data.empty:
        st.error(
            "Yahoo did not provide usable XAUUSD 15-minute candles. "
            "This can happen with free data sources."
        )
        st.warning(
            "Upload genuine XAUUSD 15-minute OHLC candles from your "
            "broker as a CSV file to continue."
        )
uploaded_file = st.file_uploader(
    "Upload TradingView or MT5 chart / CSV",
    type=["jpg", "jpeg", "png", "csv"],
    help="Upload a JPG, JPEG or PNG chart screenshot from TradingView or MT5, or a CSV file containing candle data."
)

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    if file_name.endswith((".jpg", ".jpeg", ".png")):
        st.image(
            uploaded_file,
            caption="Uploaded TradingView / MT5 chart",
            use_container_width=True
        )

    elif file_name.endswith(".csv"):
        st.success("CSV file uploaded successfully.")


    if uploaded:
        try:
            data = load_csv(uploaded)
            source_name = "Uploaded CSV"
        except Exception as error:
            st.error(str(error))


# ==========================================
# DASHBOARD
# ==========================================

if not data.empty:

    st.success("Market data loaded: " + source_name)

    data = data.tail(3000)

    last = data.iloc[-1]
    result = analyse_market(data)

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest candle close", f"{last['Close']:.2f}")
    col2.metric("HTF bias", result["bias"])
    col3.metric("15m structure", result["structure"])

    st.divider()

    st.subheader("CHAMPION GOLD BOT — Analysis")

    if result["result"] == "BUY SETUP CANDIDATE":
        st.success(result["result"])
    elif result["result"] == "SELL SETUP CANDIDATE":
        st.error(result["result"])
    else:
        st.warning(result["result"])

    st.write("**BOS / CHOCH check:**", result["bos"])
    st.write("**Liquidity check:**", result["sweep"])
    st.write("**Candle confirmation:**", result["candle"])

    if result["ob"]:
        st.write(
            "**Order block zone:**",
            f"{result['ob']['low']:.2f} — "
            f"{result['ob']['high']:.2f}"
        )
        st.write(
            "**Order block retested:**",
            "YES" if result["ob_pass"] else "NO"
        )
    else:
        st.write("**Order block:** Not identified")

    st.subheader("SMC checklist")

    for name, passed in result["checks"].items():
        if passed:
            st.write("✅", name)
        else:
            st.write("❌", name)

    st.divider()

    st.subheader("XAUUSD 15-minute chart")

    chart = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="XAUUSD"
            )
        ]
    )

    chart.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Recent candles")
    st.dataframe(data.tail(10), use_container_width=True)

    st.caption(
        "Last candle timestamp: " + str(data.index[-1])
    )

else:
    st.info(
        "Waiting for usable XAUUSD data. Select CSV upload "
        "when you have genuine gold candles."
    )

st.divider()

st.caption(
    "CHAMPION GOLD BOT | Educational SMC analysis. "
    "Simplified rules can produce false signals. "
    "BOS, CHOCH, liquidity and order-block interpretations "
    "require manual review."
)
