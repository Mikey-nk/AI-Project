import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from abc import ABC, abstractmethod
import google.generativeai as genai
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Updated to use secrets.toml for everything
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

# Authentication Configuration
AUTHORIZED_USERS = {
    "kibe5067@gmail.com": "mikey19nk"
}

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# --------------- Authentication Functions ---------------
def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user credentials using secrets.toml"""
    return email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == password


def login_page():
    """Display enhanced login page"""
    st.set_page_config(page_title="AI Trading System - Login", layout="centered")

    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Trading Agent Management System</h1>
        <h3>Secure Login Portal</h3>
        <p>Multi-Agent Trading System with Real-Time Market Data & Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### 🔐 Please Login")

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                login_button = st.form_submit_button("🚀 Login")

            if login_button:
                if email and password:
                    if authenticate_user(email, password):
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter both email and password.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Demo users display
    with st.expander("📋 Demo Users (for testing)", expanded=False):
        for email, password in AUTHORIZED_USERS.items():
            st.markdown(f"**Email:** {email}  \n**Password:** {password}")


def logout():
    """Handle user logout"""
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.cache_data.clear()
    st.rerun()


def check_authentication():
    """Check if user is authenticated"""
    if not st.session_state.get("authenticated", False):
        login_page()
        return False
    return True


# --------------- Data Classes ---------------
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


# --------------- Market Data Provider ---------------
class MarketDataProvider:
    """Enhanced market data provider with multiple sources"""

    def __init__(self):
        self.alpha_vantage = None
        if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "your_alpha_vantage_api_key_here":
            try:
                self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage: {e}")

    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock price using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                return {
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "open": float(hist['Open'].iloc[0]),
                    "high": float(hist['High'].max()),
                    "low": float(hist['Low'].min()),
                    "volume": int(hist['Volume'].sum()),
                    "market_cap": info.get('marketCap', 0),
                    "pe_ratio": info.get('trailingPE', 0),
                    "timestamp": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._generate_mock_stock_data(symbol)

    def _generate_mock_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock stock data for fallback"""
        base_prices = {
            'AAPL': 175.0, 'MSFT': 340.0, 'GOOGL': 135.0, 'AMZN': 145.0,
            'TSLA': 210.0, 'NVDA': 420.0, 'META': 315.0, 'NFLX': 450.0
        }

        base_price = base_prices.get(symbol, 100.0)
        current_price = base_price * (1 + np.random.normal(0, 0.02))
        open_price = current_price * (1 + np.random.normal(0, 0.005))

        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "open": float(open_price),
            "high": float(max(current_price, open_price) * 1.02),
            "low": float(min(current_price, open_price) * 0.98),
            "volume": int(np.random.randint(1000000, 10000000)),
            "market_cap": int(current_price * 1000000000),
            "pe_ratio": float(np.random.uniform(15, 35)),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def get_stock_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get stock data for multiple symbols"""
        market_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d", interval="1m")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = info.get('previousClose', current_price)
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        current_price=float(current_price),
                        change=float(change),
                        change_percent=float(change_percent),
                        volume=int(info.get('volume', 0)),
                        timestamp=datetime.datetime.now(),
                        high_52w=float(info.get('fiftyTwoWeekHigh', 0)),
                        low_52w=float(info.get('fiftyTwoWeekLow', 0)),
                        market_cap=int(info.get('marketCap', 0))
                    )
                else:
                    market_data[symbol] = self._generate_mock_market_data(symbol)

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                market_data[symbol] = self._generate_mock_market_data(symbol)

        return market_data

    def _generate_mock_market_data(self, symbol: str) -> MarketData:
        """Generate realistic mock market data"""
        base_prices = {
            'AAPL': 175.0, 'MSFT': 340.0, 'GOOGL': 135.0, 'AMZN': 145.0,
            'TSLA': 210.0, 'NVDA': 420.0, 'META': 315.0, 'NFLX': 450.0
        }

        base_price = base_prices.get(symbol, 100.0)
        current_price = base_price * (1 + np.random.normal(0, 0.02))
        change = np.random.normal(0, 2.0)
        change_percent = (change / current_price) * 100

        return MarketData(
            symbol=symbol,
            current_price=current_price,
            change=change,
            change_percent=change_percent,
            volume=np.random.randint(1000000, 10000000),
            timestamp=datetime.datetime.now(),
            high_52w=current_price * 1.3,
            low_52w=current_price * 0.7,
            market_cap=int(current_price * 1000000000)
        )

    def get_crypto_prices(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get cryptocurrency prices"""
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']

        crypto_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    crypto_data[symbol] = {
                        "current_price": float(hist['Close'].iloc[-1]),
                        "change_24h": float(
                            ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100)
                    }
                else:
                    base_prices = {'BTC-USD': 45000, 'ETH-USD': 3000, 'ADA-USD': 1.2}
                    base_price = base_prices.get(symbol, 100)
                    crypto_data[symbol] = {
                        "current_price": float(base_price * (1 + np.random.normal(0, 0.03))),
                        "change_24h": float(np.random.normal(0, 5))
                    }
            except Exception as e:
                logger.error(f"Error fetching crypto data for {symbol}: {e}")
                base_prices = {'BTC-USD': 45000, 'ETH-USD': 3000, 'ADA-USD': 1.2}
                base_price = base_prices.get(symbol, 100)
                crypto_data[symbol] = {
                    "current_price": float(base_price * (1 + np.random.normal(0, 0.03))),
                    "change_24h": float(np.random.normal(0, 5))
                }

        return crypto_data

    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data"""
        try:
            indices = ['^GSPC', '^DJI', '^IXIC']
            market_data = {}

            for index in indices:
                ticker = yf.Ticker(index)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100

                    market_data[index] = {
                        "current": float(current),
                        "change_percent": float(change),
                        "volume": int(hist['Volume'].iloc[-1])
                    }

            return market_data
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {
                '^GSPC': {"current": 4200.0, "change_percent": 0.5, "volume": 3000000000},
                '^DJI': {"current": 33000.0, "change_percent": -0.2, "volume": 400000000},
                '^IXIC': {"current": 13000.0, "change_percent": 1.2, "volume": 5000000000}
            }


# --------------- Gemini AI Client ---------------
class GeminiClient:
    def __init__(self):
        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini AI successfully initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Gemini API"""
        if not self.model:
            return self._fallback_response(prompt)

        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when Gemini is unavailable"""
        return f"🤖 AI Analysis: Based on the task '{prompt[:100]}...' - [Simulated response - Gemini API not available or configured. Please check your API key in secrets.toml]"


# --------------- Task Logging Decorator ---------------
def log_task_completion(fn):
    """Decorator to log task completion"""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        task = fn(self, *args, **kwargs)
        if task and hasattr(st.session_state, 'show_task_logs') and st.session_state.show_task_logs:
            with st.expander(f"📋 Task {task.id} Completed", expanded=False):
                st.write("**Description:**", task.description)
                st.write("**Status:**", task.status)
                if task.result:
                    st.write("**Result:**", task.result[:500] + "..." if len(task.result) > 500 else task.result)
        return task
    return wrapper


# --------------- Base Agent Class ---------------
class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.tasks_queue = []
        self.current_task = None
        self.gemini_client = GeminiClient()
        self.market_provider = MarketDataProvider()

    @abstractmethod
    def get_agent_prompt(self) -> str:
        """Get the specific prompt for this agent type"""
        pass

    def process_task(self, task: Task) -> str:
        """Process task using Gemini API with market data context"""
        market_context = ""
        if task.market_data:
            market_summary = []
            for symbol, data in task.market_data.items():
                if isinstance(data, MarketData):
                    market_summary.append(
                        f"{symbol}: ${data.current_price:.2f} "
                        f"({data.change_percent:+.2f}%), Volume: {data.volume:,}"
                    )
                else:
                    market_summary.append(f"{symbol}: {data}")
            market_context = f"Current Market Data:\n" + "\n".join(market_summary)

        agent_prompt = self.get_agent_prompt()
        full_prompt = f"""
{agent_prompt}

Task: {task.description}

{market_context}

Please provide a detailed analysis and recommendation based on your role as {self.name}.
"""

        self.metrics.api_calls_made += 1
        return self.gemini_client.generate_response(full_prompt, market_context)

    def add_task(self, task: Task):
        if not task.market_data:
            try:
                task.market_data = self.market_provider.get_stock_data(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
            except:
                task.market_data = self.market_provider.get_market_overview()

        self.tasks_queue.append(task)
        self.tasks_queue.sort(key=lambda x: x.priority.value, reverse=True)

    @log_task_completion
    def execute_next_task(self) -> Optional[Task]:
        if not self.tasks_queue:
            return None

        task = self.tasks_queue.pop(0)
        self.current_task = task
        self.status = AgentStatus.ACTIVE

        try:
            start_time = time.time()
            result = self.process_task(task)
            end_time = time.time()

            task.status = "Completed"
            task.completed_at = datetime.datetime.now()
            task.result = result

            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.last_activity = datetime.datetime.now()
            processing_time = end_time - start_time
            self.metrics.avg_processing_time = (
                    (self.metrics.avg_processing_time * (self.metrics.tasks_completed - 1) + processing_time)
                    / self.metrics.tasks_completed
            )
            self.metrics.success_rate = (
                    (self.metrics.tasks_completed - self.metrics.errors_count) / self.metrics.tasks_completed * 100
            )

            self.status = AgentStatus.IDLE
            self.current_task = None
            return task

        except Exception as e:
            task.status = "Failed"
            task.result = f"Error: {str(e)}"
            self.metrics.errors_count += 1
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.agent_id} failed to process task {task.id}: {e}")
            return task


# --------------- Specialized Agent Implementations ---------------
class TradingStrategistAgent(BaseAgent):
    def __init__(self):
        super().__init__("TS001", "AI Trading Strategist",
                         "Defines investment philosophy, risk tolerance, and performance goals")

    def get_agent_prompt(self) -> str:
        return """You are an AI Trading Strategist responsible for developing investment strategies.
        Your role is to:
        - Define investment philosophy and approach
        - Set risk tolerance parameters
        - Establish performance goals and benchmarks
        - Analyze market conditions and trends
        - Recommend strategic asset allocation

        Consider current market conditions, economic indicators, and risk factors when making recommendations."""


class QuantAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("QA001", "Quantitative Analyst",
                         "Develops mathematical models to identify market inefficiencies")

    def get_agent_prompt(self) -> str:
        return """You are a Quantitative Analyst specializing in mathematical modeling for trading.
        Your role is to:
        - Develop statistical models for price prediction
        - Identify market inefficiencies and arbitrage opportunities
        - Analyze correlations and patterns in market data
        - Calculate risk metrics and volatility measures
        - Create backtesting frameworks

        Use quantitative methods and provide specific numerical analysis based on the market data."""


class RiskModelerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RM001", "AI Risk Modeler",
                         "Builds models to assess potential risks and mitigate losses")

    def get_agent_prompt(self) -> str:
        return """You are an AI Risk Modeler specializing in trading risk assessment.
        Your role is to:
        - Calculate Value at Risk (VaR) and Expected Shortfall
        - Analyze portfolio concentration and correlation risks
        - Model stress testing scenarios
        - Assess market, credit, and operational risks
        - Recommend risk mitigation strategies

        Provide specific risk metrics and actionable risk management recommendations."""


class ComplianceOfficerAgent(BaseAgent):
    def __init__(self):
        super().__init__("CO001", "Compliance Officer",
                         "Ensures adherence to financial regulations and ethical practices")

    def get_agent_prompt(self) -> str:
        return """You are a Compliance Officer ensuring regulatory adherence in AI trading.
        Your role is to:
        - Monitor compliance with financial regulations (SEC, FINRA, etc.)
        - Ensure adherence to trading rules and market regulations
        - Review trades for compliance violations
        - Implement compliance controls and procedures
        - Report on regulatory compliance status

        Focus on regulatory requirements and compliance risk assessment."""


# --------------- Manager Agent ---------------
class ManagerAgent:
    def __init__(self):
        self.agents = {
            "trading_strategist": TradingStrategistAgent(),
            "quant_analyst": QuantAnalystAgent(),
            "risk_modeler": RiskModelerAgent(),
            "compliance_officer": ComplianceOfficerAgent(),
        }
        self.task_history = []
        self.market_provider = MarketDataProvider()
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "system_uptime": 0.0,
            "total_api_calls": 0
        }

    def assign_task(self, agent_key: str, description: str, priority: TaskPriority) -> str:
        if agent_key not in self.agents:
            return f"Error: Agent {agent_key} not found"

        task_id = f"TASK_{len(self.task_history):04d}"
        task = Task(
            id=task_id,
            agent_id=self.agents[agent_key].agent_id,
            description=description,
            priority=priority
        )

        self.agents[agent_key].add_task(task)
        self.task_history.append(task)
        self.system_metrics["total_tasks"] += 1

        return f"Task {task_id} assigned to {self.agents[agent_key].name}"

    def execute_all_pending_tasks(self):
        completed_tasks = []
        for agent in self.agents.values():
            while agent.tasks_queue:
                completed_task = agent.execute_next_task()
                if completed_task:
                    completed_tasks.append(completed_task)
                    if completed_task.status == "Completed":
                        self.system_metrics["completed_tasks"] += 1
                    else:
                        self.system_metrics["failed_tasks"] += 1

                    self.system_metrics["total_api_calls"] = sum(
                        agent.metrics.api_calls_made for agent in self.agents.values()
                    )
        return completed_tasks

    def get_system_status(self) -> Dict[str, Any]:
        active_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ACTIVE)
        total_pending_tasks = sum(len(agent.tasks_queue) for agent in self.agents.values())

        return {
            "active_agents": active_agents,
            "total_agents": len(self.agents),
            "pending_tasks": total_pending_tasks,
            "system_health": "Healthy" if active_agents > 0 or total_pending_tasks == 0 else "Warning",
            **self.system_metrics
        }

    def get_current_market_data(self) -> Dict[str, MarketData]:
        """Get current market data for dashboard display"""
        return self.market_provider.get_stock_data(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'])


# --------------- Helper Functions ---------------
@st.cache_resource
def get_manager():
    return ManagerAgent()


def display_market_data(market_data: Dict[str, MarketData]):
    """Display current market data in the sidebar"""
    st.subheader("📈 Live Market Data")

    for symbol, data in market_data.items():
        if isinstance(data, MarketData):
            delta_color = "normal" if data.change >= 0 else "inverse"
            st.metric(
                label=symbol,
                value=f"${data.current_price:.2f}",
                delta=f"{data.change_percent:+.2f}%",
                delta_color=delta_color
            )


# --------------- Main Application ---------------
def main():
    # Check authentication first
    if not check_authentication():
        return

    st.set_page_config(page_title="AI Trading Agent Management System", layout="wide")

    # Header with logout option
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🤖 AI Trading Agent Management System")
        st.markdown("**Multi-Agent Trading System with Real-Time Market Data & Gemini AI**")
    with col2:
        st.markdown(f"**Welcome:** {st.session_state.user_email}")
        if st.button("🚪 Logout", type="secondary"):
            logout()

    # Configuration Status Alert
    config_status = []
    if not GEMINI_API_KEY:
        config_status.append("⚠️ Gemini API key not configured")
    else:
        config_status.append("✅ Gemini AI configured")

    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "your_alpha_vantage_api_key_here":
        config_status.append("ℹ️ Alpha Vantage API optional")
    else:
        config_status.append("✅ Alpha Vantage configured")

    st.info(" | ".join(config_status))

    manager = get_manager()

    # Get current market data
    try:
        market_data = manager.get_current_market_data()
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        market_data = {}

    # Sidebar for system controls and market data
    with st.sidebar:
        st.markdown(f"**👤 Logged in as:**")
        st.info(f"{st.session_state.user_email}")

        st.header("🎛️ System Controls")

        if st.button("🔄 Refresh Market Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()

        if st.button("▶️ Execute All Pending Tasks", type="primary"):
            st.session_state.show_task_logs = True
            with st.spinner("Executing tasks with Gemini AI..."):
                completed = manager.execute_all_pending_tasks()
                st.success(f"✅ Executed {len(completed)} tasks")

        # Display market data
        if market_data:
            display_market_data(market_data)

        st.header("📊 System Status")
        status = manager.get_system_status()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Agents", status["active_agents"])
            st.metric("Pending Tasks", status["pending_tasks"])
        with col2:
            st.metric("Completed", status["completed_tasks"])
            st.metric("API Calls", status["total_api_calls"])

        st.write(f"**System Health:** {status['system_health']}")

        # Auto-refresh option
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🤖 Agent Overview", "📈 Market Data", "📋 Task Assignment", "📊 Task History"])

    with tab1:
        st.header("Agent Status Dashboard")

        # Create agent status cards
        cols = st.columns(2)
        for i, (key, agent) in enumerate(manager.agents.items()):
            with cols[i % 2]:
                status_color = {
                    AgentStatus.ACTIVE: "🟢",
                    AgentStatus.IDLE: "🟡",
                    AgentStatus.ERROR: "🔴",
                    AgentStatus.MAINTENANCE: "🟠"
                }.get(agent.status, "⚪")

                ai_status = "🤖 AI Enabled" if agent.gemini_client.model else "🔧 Fallback Mode"

                st.info(f"""
                **{status_color} {agent.name}**  
                Status: {agent.status.value}  
                {ai_status}  
                Tasks Completed: {agent.metrics.tasks_completed}  
                Success Rate: {agent.metrics.success_rate:.1f}%  
                API Calls: {agent.metrics.api_calls_made}  
                Queue: {len(agent.tasks_queue)} tasks
                """)

    with tab2:
        st.header("📈 Market Data & Analysis")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Stock Analysis")
            symbol = st.selectbox("Select Stock Symbol:",
                                  ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'])

            if st.button("📊 Get Stock Data"):
                with st.spinner(f"Fetching data for {symbol}..."):
                    stock_data = manager.market_provider.get_stock_price(symbol.upper())

                    if "error" not in stock_data:
                        # Display main metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${stock_data['current_price']:.2f}")
                        with col2:
                            daily_change = stock_data['current_price'] - stock_data['open']
                            daily_change_pct = (daily_change / stock_data['open']) * 100
                            st.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:+.2f}%")
                        with col3:
                            st.metric("Volume", f"{stock_data['volume']:,}")

                        st.subheader(f"📊 Detailed Data for {symbol.upper()}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Open:** ${stock_data['open']:.2f}")
                            st.write(f"**High:** ${stock_data['high']:.2f}")
                        with col2:
                            st.write(f"**Low:** ${stock_data['low']:.2f}")
                            st.write(f"**P/E Ratio:** {stock_data.get('pe_ratio', 0):.2f}")
                        with col3:
                            market_cap = stock_data.get('market_cap', 0)
                            if market_cap > 1e12:
                                cap_str = f"${market_cap / 1e12:.2f}T"
                            elif market_cap > 1e9:
                                cap_str = f"${market_cap / 1e9:.2f}B"
                            else:
                                cap_str = f"${market_cap / 1e6:.2f}M"
                            st.write(f"**Market Cap:** {cap_str}")

                        st.caption(f"⏱️ Last Updated: {stock_data['timestamp']}")
                    else:
                        st.error(f"❌ Error fetching data")

        with col_right:
            st.subheader("💰 Cryptocurrency Prices")
            crypto_data = manager.market_provider.get_crypto_prices()

            if crypto_data:
                for symbol, data in crypto_data.items():
                    change_color = "🟢" if data["change_24h"] >= 0 else "🔴"
                    crypto_name = symbol.replace("-USD", "")
                    st.metric(
                        label=f"{crypto_name}",
                        value=f"${data['current_price']:,.2f}",
                        delta=f"{change_color} {data['change_24h']:+.2f}%"
                    )

        # Market Overview Section
        st.subheader("🏛️ Market Indices Overview")
        market_overview = manager.market_provider.get_market_overview()

        if market_overview:
            cols = st.columns(len(market_overview))
            for i, (index, data) in enumerate(market_overview.items()):
                with cols[i]:
                    change_color = "🟢" if data["change_percent"] >= 0 else "🔴"
                    index_name = index.replace("^", "")
                    if index_name == "GSPC":
                        index_name = "S&P 500"
                    elif index_name == "DJI":
                        index_name = "Dow Jones"
                    elif index_name == "IXIC":
                        index_name = "NASDAQ"

                    st.metric(
                        index_name,
                        f"{data['current']:,.2f}",
                        f"{change_color} {data['change_percent']:+.2f}%"
                    )

    with tab3:
        st.header("📋 Task Assignment Center")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📝 Custom Task Assignment")

            selected_agent = st.selectbox(
                "Select Agent:",
                options=list(manager.agents.keys()),
                format_func=lambda x: f"{manager.agents[x].name} - {manager.agents[x].description[:50]}..."
            )

            task_description = st.text_area(
                "Task Description:",
                height=120,
                placeholder="Enter a detailed task description. Example: 'Analyze AAPL stock performance and provide investment recommendation'"
            )

            priority = st.selectbox(
                "Priority Level:",
                options=[TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL],
                format_func=lambda x: f"{x.name} - {['Routine task', 'Standard priority', 'Urgent task', 'Immediate attention'][x.value - 1]}",
                index=1
            )

            if st.button("🚀 Assign Task", type="primary", use_container_width=True):
                if task_description.strip():
                    result = manager.assign_task(selected_agent, task_description.strip(), priority)
                    st.success(f"✅ {result}")
                else:
                    st.error("❌ Please enter a task description")

        with col2:
            st.subheader("⚡ Quick Tasks")

            quick_tasks = {
                "📊 Analyze AAPL": {
                    "description": "Provide comprehensive analysis of AAPL stock including technical indicators and investment recommendation",
                    "agent": "quant_analyst"
                },
                "⚠️ Risk Assessment": {
                    "description": "Evaluate current market risk and provide risk mitigation recommendations",
                    "agent": "risk_modeler"
                },
                "🎯 Trading Strategy": {
                    "description": "Develop trading strategy based on current market conditions",
                    "agent": "trading_strategist"
                },
                "🛡️ Compliance Check": {
                    "description": "Review current strategies for regulatory compliance",
                    "agent": "compliance_officer"
                }
            }

            for task_name, task_info in quick_tasks.items():
                if st.button(task_name, key=f"quick_{task_name}", use_container_width=True):
                    result = manager.assign_task(
                        task_info["agent"],
                        task_info["description"],
                        TaskPriority.HIGH
                    )
                    st.success(f"✅ {result}")

    with tab4:
        st.header("📊 Task History & Results")

        if manager.task_history:
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "🔍 Filter by Status:",
                    ["Pending", "Completed", "Failed"],
                    default=["Completed", "Failed"]
                )
            with col2:
                agent_filter = st.multiselect(
                    "🤖 Filter by Agent:",
                    [agent.name for agent in manager.agents.values()],
                    default=[agent.name for agent in manager.agents.values()]
                )

            # Create task history dataframe
            task_data = []
            for task in manager.task_history:
                agent_name = next(
                    (agent.name for agent in manager.agents.values() if agent.agent_id == task.agent_id),
                    "Unknown"
                )

                if task.status in status_filter and agent_name in agent_filter:
                    task_data.append({
                        "Task ID": task.id,
                        "Agent": agent_name,
                        "Description": task.description[:60] + "..." if len(task.description) > 60 else task.description,
                        "Priority": task.priority.name,
                        "Status": task.status,
                        "Created": task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Completed": task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else "-",
                        "Has Result": "✅" if task.result else "❌"
                    })

            if task_data:
                df_tasks = pd.DataFrame(task_data)

                # Display task statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Filtered Tasks", len(task_data))
                with col2:
                    completed_count = len([t for t in task_data if t["Status"] == "Completed"])
                    st.metric("Completed", completed_count)
                with col3:
                    failed_count = len([t for t in task_data if t["Status"] == "Failed"])
                    st.metric("Failed", failed_count)
                with col4:
                    pending_count = len([t for t in task_data if t["Status"] == "Pending"])
                    st.metric("Pending", pending_count)

                # Display tasks table
                st.subheader("📊 Task Overview")
                st.dataframe(df_tasks, use_container_width=True, height=400)

                # Show detailed task results
                st.subheader("🔍 Detailed Task Results")
                completed_tasks = [task for task in manager.task_history
                                   if task.status == "Completed" and task.result and
                                   any(t["Task ID"] == task.id for t in task_data)]

                if completed_tasks:
                    selected_task = st.selectbox(
                        "Select Task for Detailed View:",
                        completed_tasks,
                        format_func=lambda t: f"{t.id} - {t.description[:50]}..."
                    )

                    if selected_task:
                        with st.container():
                            st.markdown(f"### 📋 Task Details: {selected_task.id}")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Agent:** {next((agent.name for agent in manager.agents.values() if agent.agent_id == selected_task.agent_id), 'Unknown')}")
                                st.write(f"**Priority:** {selected_task.priority.name}")
                                st.write(f"**Status:** {selected_task.status}")
                            with col2:
                                st.write(f"**Created:** {selected_task.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                                if selected_task.completed_at:
                                    st.write(f"**Completed:** {selected_task.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
                                    processing_time = (selected_task.completed_at - selected_task.created_at).total_seconds()
                                    st.write(f"**Processing Time:** {processing_time:.2f}s")

                            st.markdown("**📝 Task Description:**")
                            st.text_area("", selected_task.description, height=80, key="desc_readonly", disabled=True)

                            st.markdown("**🤖 AI Generated Result:**")
                            st.text_area("", selected_task.result, height=300, key="result_readonly", disabled=True)

                            if selected_task.market_data:
                                st.markdown("**📊 Associated Market Data:**")
                                if isinstance(list(selected_task.market_data.values())[0], MarketData):
                                    market_dict = {}
                                    for symbol, data in selected_task.market_data.items():
                                        market_dict[symbol] = {
                                            "current_price": f"${data.current_price:.2f}",
                                            "change_percent": f"{data.change_percent:+.2f}%",
                                            "volume": f"{data.volume:,}",
                                            "market_cap": f"${data.market_cap:,}" if data.market_cap else "N/A"
                                        }
                                    market_df = pd.DataFrame(market_dict).T
                                    st.dataframe(market_df, use_container_width=True)
                else:
                    st.info("ℹ️ No completed tasks with results match the current filters")
            else:
                st.info("🔍 No tasks match the current filters")
        else:
            st.info("📋 No task history available yet. Use the 'Task Assignment' tab to create your first task!")

    # Footer with system information
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_status = "🟢 Connected" if GEMINI_API_KEY else "🔴 Not Configured"
        st.write(f"**Gemini AI:** {gemini_status}")

    with col2:
        av_status = "🟢 Connected" if (ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "your_alpha_vantage_api_key_here") else "🟡 Optional"
        st.write(f"**Alpha Vantage:** {av_status}")

    with col3:
        st.write(f"**Market Data:** 🟢 yfinance Active")

    with col4:
        st.write(f"**Last Updated:** {datetime.datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()