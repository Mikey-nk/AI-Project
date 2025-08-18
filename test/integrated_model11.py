# integrated_model_full.py
"""
Self-contained AI trading system:
- model9_1 authentication (simple email/password via st.secrets fallback)
- Enhanced market provider and analytics (from model10 ideas)
- Gemini wrapper (fallback mode if API not set)
- Agents (Strategist, Quant, Risk, Compliance)
- Manager agent that assigns tasks and executes agents' work
- ExecutionAgent + Alpaca REST client for paper trading
- SQLite persistence for tasks and metrics
- Streamlit UI: login, dashboards, task management, trading tab

Requirements:
pip install streamlit yfinance plotly google-generativeai alpha_vantage tenacity requests pandas numpy
(only needed parts used depending on configuration)
"""
# standard libs
import time
import datetime
import logging
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from functools import wraps
from collections import defaultdict, deque

# third-party
import streamlit as st
import yfinance as yf
import numpy as np
import requests

# Optional packages that enhance functionality if present:
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:
    # fallback no-retry decorator
    def retry(*args, **kwargs):
        def inner(f): return f
        return inner

# Optional Google Gemini client. If not installed or API absent, system will fallback.
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("integrated_model_full")

# ---------------- CONFIGURATION ----------------
# Use Streamlit secrets if available; otherwise fallback to local defaults (not secure, only for dev/testing)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") if st.secrets else ""
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "") if st.secrets else ""
# AUTHORIZED_USERS expected to be dict of email:password (plain text) in secrets for model9_1 auth
AUTHORIZED_USERS = st.secrets.get("AUTHORIZED_USERS", {}) if st.secrets else {"kibe5067@gmail.com": "mikey19nk"}

# Alpaca config (paper trading)
ALPACA_API_KEY = st.secrets.get("ALPACA_API_KEY", "") if st.secrets else ""
ALPACA_SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", "") if st.secrets else ""
ALPACA_BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") if st.secrets else "https://paper-api.alpaca.markets"

# DB path
DB_PATH = "trading_system.db"

# ---------------- Authentication (model9_1 style) ----------------
def authenticate_user(email: str, password: str) -> bool:
    """
    Simple authentication using a secrets-stored map of email -> password (plain text).
    This mirrors the model9_1 approach requested by the user.
    For production: switch to hashed passwords, 2FA, or an OAuth provider.
    """
    return email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == password

def login_page():
    st.set_page_config(page_title="AI Trading System - Login", layout="centered")
    st.markdown("""
    <style>
    .main-header { text-align: center; color: #1f77b4; margin-bottom: 1rem; }
    .login-box { max-width: 420px; margin: 0 auto; padding: 1.25rem; border-radius: 8px; background: #f8f9fa; box-shadow: 0 4px 8px rgba(0,0,0,0.06); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>🤖 AI Trading Agent Management System</h1><p>Secure login (demo)</p></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("### 🔐 Please sign in")
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("📧 Email address", placeholder="you@example.com")
            password = st.text_input("🔑 Password", type="password")
            submitted = st.form_submit_button("🚀 Login")
            if submitted:
                if not email or not password:
                    st.warning("Please enter both email and password.")
                elif authenticate_user(email, password):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("✅ Login successful — redirecting...")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Demo credentials (for testing)"):
        for e, p in AUTHORIZED_USERS.items():
            st.write(f"**{e}** / {p}")

def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.success("Logged out")
    time.sleep(0.4)
    st.rerun()


def require_login() -> bool:
    if not st.session_state.get("authenticated", False):
        login_page()
        return False
    return True

# ---------------- Data classes & Enums ----------------
class AgentStatus(Enum):
    ACTIVE = "Active"
    IDLE = "Idle"
    ERROR = "Error"
    MAINTENANCE = "Maintenance"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class MarketData:
    symbol: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime.datetime
    high_52w: float = 0.0
    low_52w: float = 0.0
    market_cap: float = 0.0
    pe_ratio: float = 0.0

@dataclass
class AgentMetrics:
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_processing_time: float = 0.0
    last_activity: Optional[datetime.datetime] = None
    errors_count: int = 0
    api_calls_made: int = 0

@dataclass
class Task:
    id: str
    agent_id: str
    description: str
    priority: TaskPriority
    market_data: Optional[Dict[str, MarketData]] = None
    status: str = "Pending"
    created_at: datetime.datetime = None
    completed_at: Optional[datetime.datetime] = None
    result: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()

# ---------------- DatabaseManager (simple SQLite) ----------------
class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                description TEXT,
                priority INTEGER,
                status TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                agent_id TEXT,
                timestamp TIMESTAMP,
                tasks_completed INTEGER,
                success_rate REAL,
                avg_processing_time REAL,
                errors_count INTEGER,
                api_calls_made INTEGER
            )
            """)
            conn.commit()

    def save_task(self, task: Task):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
            INSERT OR REPLACE INTO tasks (id, agent_id, description, priority, status, created_at, completed_at, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (task.id, task.agent_id, task.description, task.priority.value, task.status, task.created_at, task.completed_at, task.result))
            conn.commit()

    def get_tasks(self, limit=100):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT id, agent_id, description, priority, status, created_at, completed_at, result FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = c.fetchall()
            return rows

db = DatabaseManager()

# ---------------- Market Data Provider ----------------
class MarketDataProvider:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 30  # seconds

    def _cache_key(self, symbol: str):
        return f"md_{symbol}"

    def _get_cached(self, key):
        v = self.cache.get(key)
        if not v:
            return None
        data, ts = v
        if (datetime.datetime.now() - ts).total_seconds() < self.cache_ttl:
            return data
        del self.cache[key]
        return None

    def _set_cache(self, key, data):
        self.cache[key] = (data, datetime.datetime.now())

    def get_stock_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        result = {}
        for s in symbols:
            key = self._cache_key(s)
            cached = self._get_cached(key)
            if cached:
                result[s] = cached
                continue
            try:
                ticker = yf.Ticker(s)
                info = ticker.info if hasattr(ticker, "info") else {}
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_close = float(info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price))
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
                    md = MarketData(
                        symbol=s,
                        current_price=current_price,
                        change=change,
                        change_percent=change_pct,
                        volume=int(info.get('volume', hist['Volume'].iloc[-1] if 'Volume' in hist else 0)),
                        timestamp=datetime.datetime.now(),
                        high_52w=float(info.get("fiftyTwoWeekHigh", 0.0)),
                        low_52w=float(info.get("fiftyTwoWeekLow", 0.0)),
                        market_cap=int(info.get("marketCap", 0)) if info.get("marketCap") else 0,
                        pe_ratio=float(info.get("trailingPE", 0.0)) if info.get("trailingPE") else 0.0
                    )
                else:
                    # fallback simulation
                    base = 100.0
                    current_price = float(base * (1 + np.random.normal(0, .02)))
                    change = float(np.random.normal(0, 1.0))
                    change_pct = (change / current_price) * 100
                    md = MarketData(symbol=s, current_price=current_price, change=change, change_percent=change_pct, volume=int(np.random.randint(100000, 1000000)), timestamp=datetime.datetime.now())
            except Exception as e:
                logger.warning(f"Market data fetch error for {s}: {e}")
                base = 100.0
                current_price = float(base * (1 + np.random.normal(0, .02)))
                change = float(np.random.normal(0, 1.0))
                change_pct = (change / current_price) * 100
                md = MarketData(symbol=s, current_price=current_price, change=change, change_percent=change_pct, volume=int(np.random.randint(100000, 1000000)), timestamp=datetime.datetime.now())
            result[s] = md
            self._set_cache(key, md)
        return result

market_provider = MarketDataProvider()

# ---------------- Gemini wrapper (light) ----------------
class GeminiClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.enabled = False
        self.model = None
        if genai and api_key:
            try:
                genai.configure(api_key=api_key)
                # use default model
                self.model = genai.GenerativeModel("gemini-2.5-flash") if hasattr(genai, "GenerativeModel") else None
                self.enabled = True
                logger.info("Gemini configured")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")
                self.enabled = False

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.enabled or not self.model:
            # fallback simulated reply
            return f"[SIMULATED] Analysis for: {prompt[:200]} ... (enable Gemini to get real responses)"
        try:
            resp = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens))
            return getattr(resp, "text", str(resp))
        except Exception as e:
            logger.error(f"Gemini generate error: {e}")
            return f"[SIMULATED on error] {prompt[:200]}"

gemini_client = GeminiClient(GEMINI_API_KEY)

# ---------------- Base agent & specialized agents ----------------
def log_task(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        task = fn(self, *args, **kwargs)
        # minimal logging; UI displays tasks elsewhere
        if task:
            logger.info(f"Task {task.id} processed by {self.name} | status: {task.status}")
        return task
    return wrapper

class BaseAgent:
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.tasks_queue: List[Task] = []
        self.current_task: Optional[Task] = None

    def get_agent_prompt(self) -> str:
        return "You are an AI trading assistant."

    def augment_with_market(self, task: Task, symbols: Optional[List[str]] = None):
        if task.market_data:
            return
        syms = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN"]
        task.market_data = market_provider.get_stock_data(syms)

    def process_task(self, task: Task) -> str:
        # Create a prompt from agent's role and market data
        market_summary = ""
        if task.market_data:
            lines = []
            for s, md in task.market_data.items():
                lines.append(f"{s}: ${md.current_price:.2f} ({md.change_percent:+.2f}%) vol:{md.volume:,}")
            market_summary = "\n".join(lines)

        prompt = f"{self.get_agent_prompt()}\n\nTask: {task.description}\n\nMarket Data:\n{market_summary}\n\nProvide an analysis and recommended action."

        # call Gemini (or fallback)
        resp = gemini_client.generate(prompt, max_tokens=512)
        return resp

    def add_task(self, task: Task, symbols: Optional[List[str]] = None):
        if not task.market_data:
            self.augment_with_market(task, symbols)
        self.tasks_queue.append(task)
        self.tasks_queue.sort(key=lambda t: t.priority.value, reverse=True)

    @log_task
    def execute_next(self) -> Optional[Task]:
        if not self.tasks_queue:
            return None
        task = self.tasks_queue.pop(0)
        self.current_task = task
        self.status = AgentStatus.ACTIVE
        start = time.time()
        try:
            result = self.process_task(task)
            end = time.time()
            task.status = "Completed"
            task.completed_at = datetime.datetime.now()
            task.result = result
            # update metrics
            self.metrics.tasks_completed += 1
            proc = end - start
            self.metrics.avg_processing_time = ((self.metrics.avg_processing_time * (self.metrics.tasks_completed - 1) + proc) / (self.metrics.tasks_completed)) if self.metrics.tasks_completed else proc
            self.metrics.last_activity = datetime.datetime.now()
            self.metrics.api_calls_made += 1
            self.status = AgentStatus.IDLE
            self.current_task = None
            # persist
            db.save_task(task)
            return task
        except Exception as e:
            logger.exception("Agent processing error")
            task.status = "Failed"
            task.result = f"Error: {e}"
            self.metrics.errors_count += 1
            self.status = AgentStatus.ERROR
            db.save_task(task)
            return task

# Specialized agents
class TradingStrategistAgent(BaseAgent):
    def __init__(self):
        super().__init__("TS001", "Trading Strategist", "High-level strategy and asset allocation.")

    def get_agent_prompt(self) -> str:
        return """You are an AI Trading Strategist. Produce strategy recommendations with risk controls, sizing, and timeframes."""

class QuantAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("QA001", "Quant Analyst", "Quantitative modelling, signals and metrics.")

    def get_agent_prompt(self) -> str:
        return """You are a Quantitative Analyst. Provide model-driven, numeric analysis and clear signal rules."""

class RiskModelerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RM001", "Risk Modeler", "Compute risk metrics and mitigation steps.")

    def get_agent_prompt(self) -> str:
        return """You are a Risk Modeler. Provide VaR, drawdown, position sizing, and stress tests."""

class ComplianceOfficerAgent(BaseAgent):
    def __init__(self):
        super().__init__("CO001", "Compliance Officer", "Check for regulatory and policy compliance.")

    def get_agent_prompt(self) -> str:
        return """You are a Compliance Officer. Validate trades & strategies against regulatory rules and internal policies."""

# ---------------- ManagerAgent (orchestrates agents) ----------------
class ManagerAgent:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            "strategist": TradingStrategistAgent(),
            "quant": QuantAnalystAgent(),
            "risk": RiskModelerAgent(),
            "compliance": ComplianceOfficerAgent(),
        }
        self.task_history: List[Task] = []
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_api_calls": 0
        }

    def assign_task(self, agent_key: str, description: str, priority: TaskPriority, symbols: Optional[List[str]] = None) -> str:
        if agent_key not in self.agents:
            return f"Agent {agent_key} not found"
        task_id = f"TASK_{len(self.task_history) + 1:04d}"
        task = Task(id=task_id, agent_id=self.agents[agent_key].agent_id, description=description, priority=priority)
        self.agents[agent_key].add_task(task, symbols)
        self.task_history.append(task)
        self.system_metrics["total_tasks"] += 1
        db.save_task(task)
        return f"Assigned {task_id} to {agent_key}"

    def execute_all_pending_tasks(self) -> List[Task]:
        completed = []
        for a in self.agents.values():
            while a.tasks_queue:
                t = a.execute_next()
                if t:
                    completed.append(t)
                    if t.status == "Completed":
                        self.system_metrics["completed_tasks"] += 1
                    else:
                        self.system_metrics["failed_tasks"] += 1
        # update total API calls
        self.system_metrics["total_api_calls"] = sum(a.metrics.api_calls_made for a in self.agents.values())
        return completed

    def get_system_status(self) -> Dict[str, Any]:
        active = sum(1 for a in self.agents.values() if a.status == AgentStatus.ACTIVE)
        pending = sum(len(a.tasks_queue) for a in self.agents.values())
        return {
            "active_agents": active,
            "total_agents": len(self.agents),
            "pending_tasks": pending,
            **self.system_metrics
        }

manager = ManagerAgent()

# ---------------- Alpaca client and ExecutionAgent ----------------
class AlpacaError(Exception):
    pass

@dataclass
class OrderResultLite:
    id: str = ""
    status: str = ""
    symbol: str = ""
    qty: str = ""
    side: str = ""
    type: str = ""
    time_in_force: str = ""
    filled_qty: str = ""
    created_at: str = ""

class AlpacaClient:
    def __init__(self, api_key: str, secret: str, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json"
        })

    def _handle(self, resp: requests.Response):
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            raise AlpacaError("Non-JSON response")
        if not resp.ok:
            raise AlpacaError(data.get("message") or f"HTTP {resp.status_code}: {data}")
        return data

    def get_account(self):
        r = self.session.get(f"{self.base}/v2/account", timeout=20)
        return self._handle(r)

    def get_positions(self):
        r = self.session.get(f"{self.base}/v2/positions", timeout=20)
        return self._handle(r)

    def place_order(self, symbol: str, qty: int, side: str, type_: str = "market", time_in_force: str = "day"):
        payload = {
            "symbol": symbol.upper(),
            "qty": str(int(qty)),
            "side": side.lower(),
            "type": type_.lower(),
            "time_in_force": time_in_force.lower()
        }
        r = self.session.post(f"{self.base}/v2/orders", json=payload, timeout=30)
        data = self._handle(r)
        return OrderResultLite(
            id=data.get("id", ""),
            status=data.get("status", ""),
            symbol=data.get("symbol", ""),
            qty=data.get("qty", ""),
            side=data.get("side", ""),
            type=data.get("type", ""),
            time_in_force=data.get("time_in_force", ""),
            filled_qty=data.get("filled_qty", ""),
            created_at=data.get("created_at", "")
        )

    def cancel_order(self, order_id: str):
        r = self.session.delete(f"{self.base}/v2/orders/{order_id}", timeout=20)
        if r.status_code == 204:
            return {"status": "canceled", "id": order_id}
        return self._handle(r)

class ExecutionAgent:
    def __init__(self, client: Optional[AlpacaClient] = None):
        self.client = client

    @property
    def ready(self) -> bool:
        return self.client is not None

    def submit_signal(self, symbol: str, action: str, qty: int, tif: str = "day") -> Dict[str, Any]:
        if not self.ready:
            return {"ok": False, "error": "Broker not configured."}
        try:
            res = self.client.place_order(symbol, qty, action, "market", tif)
            return {"ok": True, "order": res}
        except Exception as e:
            logger.exception("Order placement failed")
            return {"ok": False, "error": str(e)}

    def account_overview(self) -> Dict[str, Any]:
        if not self.ready:
            return {"ok": False, "error": "Broker not configured."}
        try:
            acc = self.client.get_account()
            return {"ok": True, "account": acc}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def positions(self) -> Dict[str, Any]:
        if not self.ready:
            return {"ok": False, "error": "Broker not configured."}
        try:
            pos = self.client.get_positions()
            return {"ok": True, "positions": pos}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if not self.ready:
            return {"ok": False, "error": "Broker not configured."}
        try:
            res = self.client.cancel_order(order_id)
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

# create execution agent singleton (cached)
@st.cache_resource
def get_execution_agent() -> ExecutionAgent:
    if ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL:
        client = AlpacaClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
        return ExecutionAgent(client)
    return ExecutionAgent(None)

# ---------------- Streamlit UI ----------------
def display_system_status():
    st.sidebar.markdown("### System Status")
    status = manager.get_system_status()
    st.sidebar.metric("Active Agents", status["active_agents"])
    st.sidebar.metric("Pending Tasks", status["pending_tasks"])
    st.sidebar.metric("Total Tasks", status["total_tasks"])
    st.sidebar.metric("Completed", status["completed_tasks"])

def display_agents_overview():
    st.header("🤖 Agent Overview")
    cols = st.columns(2)
    for i, agent in enumerate(manager.agents.values()):
        with cols[i % 2]:
            st.subheader(agent.name)
            st.write(agent.description)
            st.write(f"Status: {agent.status.value}")
            st.write(f"Queue: {len(agent.tasks_queue)}")
            st.write(f"Tasks completed: {agent.metrics.tasks_completed}")
            if agent.current_task:
                st.markdown(f"**Current task:** {agent.current_task.id} — {agent.current_task.description[:120]}")

def display_market_snapshot(symbols: List[str]):
    st.header("📈 Market Snapshot")
    d = market_provider.get_stock_data(symbols)
    for s, md in d.items():
        st.metric(label=s, value=f"${md.current_price:.2f}", delta=f"{md.change_percent:+.2f}%")

def display_task_history():
    st.header("📚 Task History")
    rows = db.get_tasks(200)
    if not rows:
        st.info("No tasks saved yet.")
        return
    # present as simple table
    import pandas as pd
    df = pd.DataFrame(rows, columns=["id","agent_id","description","priority","status","created_at","completed_at","result"])
    st.dataframe(df, use_container_width=True)

def main_app():
    if not require_login():
        return

    st.set_page_config(page_title="AI Trading Agent System", layout="wide")
    # Header
    left, middle, right = st.columns([3, 1, 1])
    with left:
        st.title("🤖 AI Trading Agent Management")
        st.markdown("Multi-agent analysis + paper trading demo")
    with middle:
        st.write(f"**User:** {st.session_state.user_email}")
    with right:
        if st.button("Logout"):
            logout()

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Execute all pending tasks"):
        with st.spinner("Executing..."):
            completed = manager.execute_all_pending_tasks()
            st.success(f"Executed {len(completed)} tasks")

    if st.sidebar.button("Refresh market cache"):
        market_provider.cache.clear()
        st.success("Market cache cleared")

    display_system_status()

    tabs = st.tabs(["Dashboard", "Task Management", "Market", "Trading", "History", "System"])
    with tabs[0]:
        display_agents_overview()
        # quick assign form
        st.markdown("---")
        st.subheader("Quick Task Assign")
        with st.form("quick_task"):
            agent_key = st.selectbox("Agent", list(manager.agents.keys()))
            desc = st.text_area("Task description", "Analyze momentum and give a buy/sell signal for AAPL")
            pr = st.selectbox("Priority", list(TaskPriority))
            syms = st.text_input("Symbols (comma separated)", "AAPL,MSFT,GOOGL")
            submit = st.form_submit_button("Assign Task")
            if submit:
                syml = [s.strip().upper() for s in syms.split(",") if s.strip()]
                res = manager.assign_task(agent_key, desc, pr, syml)
                st.success(res)

    with tabs[1]:
        st.header("Task Queues per Agent")
        for key, ag in manager.agents.items():
            st.subheader(f"{ag.name} ({key})")
            for t in ag.tasks_queue:
                st.markdown(f"- **{t.id}** | {t.description[:150]} | {t.priority.name}")
            if not ag.tasks_queue:
                st.info("No queued tasks.")

    with tabs[2]:
        st.header("Market Intelligence")
        syms = st.text_input("Symbols for snapshot", "AAPL,MSFT,GOOGL,AMZN")
        syml = [s.strip().upper() for s in syms.split(",") if s.strip()]
        display_market_snapshot(syml)

    with tabs[3]:
        st.header("Trading (Paper)")
        exec_agent = get_execution_agent()
        if not exec_agent.ready:
            st.warning("Broker not configured. Provide Alpaca keys in st.secrets to enable paper trading.")
        else:
            # account
            acct = exec_agent.account_overview()
            if acct.get("ok"):
                a = acct["account"]
                col1, col2, col3, col4 = st.columns(4)
                def safef(x):
                    try: return float(x)
                    except: return 0.0
                col1.metric("Equity", f"${safef(a.get('equity')):.2f}")
                col2.metric("Cash", f"${safef(a.get('cash')):.2f}")
                col3.metric("Buying Power", f"${safef(a.get('buying_power')):.2f}")
                col4.metric("Portfolio Value", f"${safef(a.get('portfolio_value')):.2f}")
                with st.expander("Account JSON"): st.json(a)
            else:
                st.warning(acct.get("error"))

            # positions
            pos = exec_agent.positions()
            st.subheader("Open Positions")
            if pos.get("ok") and pos["positions"]:
                import pandas as pd
                rows = pos["positions"]
                df = pd.DataFrame(rows)
                useful_cols = [c for c in ["symbol","qty","avg_entry_price","market_value","unrealized_pl","unrealized_plpc"] if c in df.columns]
                if useful_cols:
                    st.dataframe(df[useful_cols], use_container_width=True)
                else:
                    st.write(df)
            else:
                st.info("No open positions or error: " + str(pos.get("error", "")))

            st.markdown("---")
            st.subheader("Place Market Order (Paper)")
            with st.form("order_form"):
                symbol = st.text_input("Symbol", "AAPL")
                side = st.selectbox("Side", ["buy", "sell"])
                qty = st.number_input("Quantity", min_value=1, value=1, step=1)
                tif = st.selectbox("Time In Force", ["day", "gtc"])
                go = st.form_submit_button("Submit Order")
                if go:
                    with st.spinner("Submitting order..."):
                        r = exec_agent.submit_signal(symbol, side, int(qty), tif)
                        if r.get("ok"):
                            st.success(f"Order placed: {r['order'].status} (id: {r['order'].id})")
                            st.json(r['order'].__dict__)
                        else:
                            st.error("Order failed: " + str(r.get("error")))

    with tabs[4]:
        display_task_history()

    with tabs[5]:
        st.header("System & Diagnostics")
        st.write("Gemini configured:", bool(gemini_client.enabled))
        st.write("Alpaca configured:", get_execution_agent().ready)
        st.write("DB path:", DB_PATH)
        st.write("Cached market keys:", list(market_provider.cache.keys()))
        if st.button("Clear all caches"):
            market_provider.cache.clear()
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.success("Caches cleared")

# run
if __name__ == "__main__":
    main_app()
