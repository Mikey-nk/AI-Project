"""
integrated_model_python_binance_full.py

AI Trading Agent Dashboard using python-binance for Spot Trading,
with Technical Charts & Portfolio P&L, Agents, SQLite, and Streamlit UI.
"""

import streamlit as st
st.set_page_config(page_title="AI Trading Agent System (Binance)", layout="wide")

import time
import datetime
import logging
import sqlite3
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import plotly.graph_objects as go
import yfinance as yf

# Optional Gemini wrapper (if API key provided)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Binance Python API
from binance.client import Client
from binance.enums import *

# -------------- Config & Secrets --------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") if st.secrets else ""
AUTHORIZED_USERS = st.secrets.get("AUTHORIZED_USERS", {}) if st.secrets else {"kibe5067@gmail.com": "mikey19nk"}
BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", "") if st.secrets else ""
BINANCE_API_SECRET = st.secrets.get("BINANCE_API_SECRET", "") if st.secrets else ""
BINANCE_TESTNET = st.secrets.get("BINANCE_TESTNET", True) if st.secrets else True

DB_PATH = "trading_system.db"

# -------------- Logging --------------
logger = logging.getLogger("ai_binance")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

# -------------- Authentication --------------
def authenticate_user(email: str, pwd: str) -> bool:
    return email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == pwd

def login_page():
    st.markdown("""
    <style>.header{ text-align:center; color:#1f77b4; margin-bottom:1rem;}
    .box{max-width:400px; margin:0 auto; padding:1rem; background:#f8f9fa; border-radius:8px;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="header"><h2>AI Trading Agent System</h2><p>Login</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="box">', unsafe_allow_html=True)
    with st.form("login"):
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if authenticate_user(email, pwd):
                st.session_state.authenticated = True
                st.session_state.user = email
                st.success("Logged in!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Demo credentials"):
        for e,p in AUTHORIZED_USERS.items():
            st.write(f"**{e}** / {p}")

def logout():
    st.session_state.clear()
    st.success("Logged out")
    time.sleep(0.3)
    st.rerun()

def require_login():
    if not st.session_state.get("authenticated", False):
        login_page()
        return False
    return True

# -------------- Data classes & DB --------------
class AgentStatus(Enum):
    IDLE = "Idle"
    ACTIVE = "Active"
    ERROR = "Error"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    agent_id: str
    description: str
    priority: TaskPriority
    result: str = ""
    status: str = "Pending"
    created_at: datetime.datetime = datetime.datetime.now()
    completed_at: datetime.datetime = None

class DBManager:
    def __init__(self, path):
        self.path = path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tasks(
                       id TEXT PRIMARY KEY, agent_id TEXT, description TEXT,
                       priority INTEGER, status TEXT, created_at TIMESTAMP, completed_at TIMESTAMP, result TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS metrics(
                       agent_id TEXT, timestamp TIMESTAMP, tasks_completed INTEGER)""")
            c.commit()

    def save_task(self, task: Task):
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                      (task.id, task.agent_id, task.description,
                       task.priority.value, task.status,
                       task.created_at, task.completed_at, task.result))
            c.commit()

    def load_tasks(self, limit=100):
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

db = DBManager(DB_PATH)

# -------------- Market & Agents (simplified/placeholder) --------------
class MarketProvider:
    def get_stock(self, symbol):
        df = yf.Ticker(symbol).history(period="1d")
        return df

class GeminiAI:
    def __init__(self, key):
        self.enabled = bool(key and genai)
        if self.enabled:
            genai.configure(api_key=key)
    def generate(self, prompt):
        return "[AI response simulated]" if not self.enabled else "AI result"

# Stub Agents
class Agent:
    def __init__(self, name):
        self.name = name; self.status = AgentStatus.IDLE

    def add_task(self, task): pass
    def process_all(self): return []

agents = {"strategist": Agent("Strategist"), "quant": Agent("Quant")}

# -------------- Binance Integration --------------
@st.cache_resource
def get_binance_client():
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        return Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=bool(BINANCE_TESTNET))
    return None

class ExecutionAgent:
    def __init__(self, client):
        self.client = client

    @property
    def ready(self):
        return self.client is not None

    def latest_price(self, symbol):
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)["price"])
        except:
            return None

    def stats_24h(self, symbol):
        try:
            return self.client.get_ticker(symbol=symbol)
        except:
            return {}

    def create_order(self, symbol, side, quantity=None, quote_qty=None):
        if not self.ready:
            return {"ok":False, "error":"Not configured"}
        try:
            params = {"symbol":symbol, "side":side, "type":ORDER_TYPE_MARKET}
            if quote_qty:
                params["quoteOrderQty"] = quote_qty
            else:
                params["quantity"] = quantity
            resp = self.client.create_order(**params)
            return {"ok":True, "order":resp}
        except Exception as e:
            return {"ok":False, "error":str(e)}

    def account(self):
        if not self.ready:
            return {"ok":False, "error":"Not configured"}
        try:
            return {"ok":True, "account":self.client.get_account()}
        except Exception as e:
            return {"ok":False, "error":str(e)}

    def balances(self):
        acc = self.account()
        if acc["ok"]:
            return {"ok":True, "balances":acc["account"].get("balances", [])}
        return acc

    def open_orders(self):
        if not self.ready:
            return {"ok":False, "error":"Not configured"}
        try:
            return {"ok":True, "orders":self.client.get_open_orders()}
        except Exception as e:
            return {"ok":False, "error":str(e)}

exec_agent = ExecutionAgent(get_binance_client())

# -------------- Technical Indicators --------------
def sma(s, w): return s.rolling(window=w).mean()
def ema(s, w): return s.ewm(span=w, adjust=False).mean()
def rsi(s, p=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    return 100 - 100/(1 + up.rolling(p).mean()/dn.rolling(p).mean())
def macd(s, f=12, s_s=26, sig=9):
    e1=ema(s, f); e2=ema(s, s_s); m=e1-e2
    sigl=ema(m, sig)
    return pd.DataFrame({"macd":m, "signal":sigl, "hist":m-sigl})

# -------------- UI Functions (Charts / Portfolio) --------------
def charts_tab():
    st.header("Charts with Overlays")
    c1, c2 = st.columns(2)
    with c1:
        sym = st.text_input("Symbol", "AAPL").upper()
        period = st.selectbox("Period", ["1mo","3mo","6mo"], index=0)
        interval = st.selectbox("Interval", ["1d","1h"], index=0)
    with c2:
        use_sma = st.checkbox("SMA", True)
        use_ema = st.checkbox("EMA", False)
    if st.button("Render"):
        df = yf.Ticker(sym).history(period=period, interval=interval)
        if df.empty:
            st.error("No data")
            return
        fig=go.Figure(data=[go.Candlestick(x=df.index,
                                           open=df['Open'], high=df['High'],
                                           low=df['Low'], close=df['Close'])])
        if use_sma:
            fig.add_trace(go.Line(x=df.index, y=sma(df['Close'], 20), name="SMA20"))
        if use_ema:
            fig.add_trace(go.Line(x=df.index, y=ema(df['Close'], 20), name="EMA20"))
        st.plotly_chart(fig, use_container_width=True)

def portfolio_tab():
    st.header("Portfolio & P&L")
    bal = exec_agent.balances()
    if not bal["ok"]:
        st.warning(bal.get("error", "No data"))
        return
    df = pd.DataFrame(bal["balances"])
    df["free"] = df["free"].astype(float)
    df["locked"] = df["locked"].astype(float)
    df["qty"] = df["free"] + df["locked"]
    df = df[df["qty"]>0]
    vals, changes = [], []
    for _, r in df.iterrows():
        asset = r["asset"]
        if asset=="USDT":
            p, c = 1.0, 0.0
        else:
            s = asset+"USDT"
            p = exec_agent.latest_price(s) or 0
            info = exec_agent.stats_24h(s)
            c = float(info.get("priceChangePercent", 0))
        vals.append(p); changes.append(c)
    df["price"] = vals
    df["value"] = df["qty"]*df["price"]
    df["24h%"] = changes
    total = df["value"].sum()
    delta = (df["value"]*(df["24h%"]/100)).sum()
    col1, col2 = st.columns(2)
    col1.metric("Portfolio Value (USDT)", f"{total:.2f}", delta=f"{(delta/total*100 if total else 0):+.2f}%")
    col2.metric("24h Change (USDT)", f"{delta:.2f}")
    st.dataframe(df[["asset","qty","price","value","24h%"]], use_container_width=True)

# -------------- Main App --------------
def app():
    if not require_login():
        return
    st.title("AI Trading Dashboard")
    if st.button("Logout"):
        logout()
    tabs = st.tabs(["Dashboard", "Charts", "Portfolio", "Trading"])
    with tabs[1]: charts_tab()
    with tabs[2]: portfolio_tab()
    with tabs[3]:
        st.header("Place Market Order")
        sym = st.text_input("Symbol", "BTCUSDT").upper()
        side = st.selectbox("Side", ["BUY","SELL"])
        mode = st.radio("By", ["Quantity","Quote USDT"])
        qty= None; quo=None
        if mode=="Quantity":
            qty = st.number_input("Quantity", value=0.001, step=0.001)
        else:
            quo = st.number_input("Spend USDT", value=10.0, step=1.0)
        if st.button("Submit"):
            res = exec_agent.create_order(sym, side, quantity=qty, quote_qty=quo)
            st.write(res)
    with tabs[0]:
        st.write("Agent placeholders (tasks, strategy) would go here.")

if __name__ == "__main__":
    app()
