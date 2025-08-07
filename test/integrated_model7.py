import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import time
import asyncio
import aiohttp
import requests
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from abc import ABC, abstractmethod
import logging
import google.generativeai as genai
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

# Authentication Configuration
AUTHORIZED_USERS = {
    "kibe50067@gmail.com": "mikey13nk"
}

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user credentials"""
    return email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == password


def login_page():
    """Display login page"""
    st.set_page_config(page_title="AI Trading System - Login", layout="centered")

    # Custom CSS for better styling
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
        font-size: 1rem;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Trading Agent Management System</h1>
        <h3>Secure Login Portal</h3>
        <p>Multi-Agent Trading System with Real-Time Market Data & Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Login form container
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        st.markdown("### 🔐 Please Login")

        # Login form
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("📧 Email Address", placeholder="Enter your email address")
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

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p><small>🔒 Secure Access Required</small></p>
            <p><small>Powered by Streamlit & Gemini AI</small></p>
        </div>
        """, unsafe_allow_html=True)


def logout():
    """Handle user logout"""
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.rerun()


def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_page()
        return False

    return True


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


class MarketDataProvider:
    """Enhanced market data provider with multiple sources and comprehensive data"""

    def __init__(self):
        self.alpha_vantage = None
        if ALPHA_VANTAGE_API_KEY:
            self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock price using yfinance (restored from model4)"""
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
            return {"error": str(e)}

    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data (restored from model4)"""
        try:
            # Major indices
            indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
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
            return {"error": str(e)}

    def get_crypto_prices(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get cryptocurrency prices (restored from model4)"""
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
            except Exception as e:
                logger.error(f"Error fetching crypto data for {symbol}: {e}")

        return crypto_data

    def get_stock_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Enhanced stock data method for model5 compatibility"""
        market_data = {}

        for symbol in symbols:
            try:
                # Primary: Yahoo Finance (free, no API key required)
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
                    # Fallback to mock data if real data unavailable
                    market_data[symbol] = self._generate_mock_data(symbol)

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                # Generate mock data as fallback
                market_data[symbol] = self._generate_mock_data(symbol)

        return market_data

    def _generate_mock_data(self, symbol: str) -> MarketData:
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

    def get_market_dashboard_data(self):
        """Get comprehensive market data for dashboard (restored from model4)"""
        return {
            "market_overview": self.get_market_overview(),
            "crypto_prices": self.get_crypto_prices(),
            "timestamp": datetime.datetime.now().isoformat()
        }


class GeminiClient:
    def __init__(self):
        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
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
        return f"AI Analysis: {prompt[:100]}... [Simulated response - Gemini API not available]"


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
                    # Handle dictionary format for backward compatibility
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
        # Fetch current market data for the task
        if not task.market_data:
            # Get both new format (MarketData objects) and old format (dict) for compatibility
            try:
                task.market_data = self.market_provider.get_stock_data(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
            except:
                # Fallback to overview data
                task.market_data = self.market_provider.get_market_overview()

        self.tasks_queue.append(task)
        self.tasks_queue.sort(key=lambda x: x.priority.value, reverse=True)

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


# Specialized Agent Implementations with Gemini Integration
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


class MLEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__("MLE001", "Machine Learning Engineer",
                         "Transforms models into scalable, production-ready software")

    def get_agent_prompt(self) -> str:
        return """You are a Machine Learning Engineer focused on production trading systems.
        Your role is to:
        - Design scalable ML pipelines for real-time trading
        - Optimize model performance and latency
        - Implement feature engineering for market data
        - Ensure system reliability and monitoring
        - Handle data preprocessing and model deployment

        Focus on technical implementation details and system architecture."""


class DataEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__("DE001", "Data Engineer",
                         "Designs and maintains data infrastructure and ETL processes")

    def get_agent_prompt(self) -> str:
        return """You are a Data Engineer responsible for market data infrastructure.
        Your role is to:
        - Design ETL pipelines for market data ingestion
        - Ensure data quality and integrity
        - Optimize data storage and retrieval systems
        - Manage real-time data feeds and APIs
        - Monitor data pipeline performance

        Focus on data architecture, quality, and system reliability."""


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


class AuditorAgent(BaseAgent):
    def __init__(self):
        super().__init__("AU001", "AI Auditor",
                         "Examines decision-making processes and ensures transparency")

    def get_agent_prompt(self) -> str:
        return """You are an AI Auditor responsible for system transparency and accountability.
        Your role is to:
        - Review AI decision-making processes
        - Ensure algorithmic transparency and explainability
        - Verify compliance with trading regulations
        - Audit system logs and trade decisions
        - Identify potential bias or errors in models

        Focus on governance, transparency, and regulatory compliance."""


class TranslatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("TR001", "AI Translator",
                         "Bridges technical AI systems and non-technical stakeholders")

    def get_agent_prompt(self) -> str:
        return """You are an AI Translator responsible for communicating complex analysis to stakeholders.
        Your role is to:
        - Translate technical analysis into business language
        - Create executive summaries and reports
        - Explain AI recommendations in simple terms
        - Provide actionable insights for decision makers
        - Communicate risks and opportunities clearly

        Focus on clear, non-technical communication that drives business decisions."""


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


class TrustAuthenticatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("TA001", "Trust Authenticator",
                         "Verifies factual accuracy and fairness of AI outputs")

    def get_agent_prompt(self) -> str:
        return """You are a Trust Authenticator responsible for verifying AI output accuracy.
        Your role is to:
        - Verify factual accuracy of AI recommendations
        - Check for bias in algorithmic decisions
        - Validate data sources and calculations
        - Ensure fairness in trading recommendations
        - Authenticate the reliability of AI outputs

        Focus on accuracy, bias detection, and trustworthiness of AI systems."""


class EthicistAgent(BaseAgent):
    def __init__(self):
        super().__init__("ET001", "AI Ethicist",
                         "Develops ethical guidelines and ensures fair, transparent decisions")

    def get_agent_prompt(self) -> str:
        return """You are an AI Ethicist ensuring ethical trading practices.
        Your role is to:
        - Develop ethical guidelines for AI trading
        - Assess the fairness and transparency of trading decisions
        - Identify potential ethical conflicts or issues
        - Ensure responsible AI deployment
        - Balance profit objectives with ethical considerations

        Focus on ethical implications and responsible AI practices."""


class LegalGuarantorAgent(BaseAgent):
    def __init__(self):
        super().__init__("LG001", "Legal Guarantor",
                         "Takes legal responsibility for AI actions and decisions")

    def get_agent_prompt(self) -> str:
        return """You are a Legal Guarantor providing legal oversight for AI trading decisions.
        Your role is to:
        - Assess legal liability of AI trading decisions
        - Ensure compliance with securities laws
        - Review contracts and legal agreements
        - Manage legal risk from AI operations
        - Provide legal accountability framework

        Focus on legal risk assessment and liability management."""


class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__("AR001", "AI Agent Architect",
                         "Designs system architecture and integrations")

    def get_agent_prompt(self) -> str:
        return """You are an AI Agent Architect designing trading system architecture.
        Your role is to:
        - Design scalable system architecture
        - Plan integration with external data sources and trading platforms
        - Ensure system security and reliability
        - Optimize system performance and throughput
        - Coordinate between different system components

        Focus on technical architecture, scalability, and system integration."""


class ManagerAgent:
    def __init__(self):
        self.agents = {
            "trading_strategist": TradingStrategistAgent(),
            "quant_analyst": QuantAnalystAgent(),
            "ml_engineer": MLEngineerAgent(),
            "data_engineer": DataEngineerAgent(),
            "risk_modeler": RiskModelerAgent(),
            "auditor": AuditorAgent(),
            "translator": TranslatorAgent(),
            "compliance_officer": ComplianceOfficerAgent(),
            "trust_authenticator": TrustAuthenticatorAgent(),
            "ethicist": EthicistAgent(),
            "legal_guarantor": LegalGuarantorAgent(),
            "architect": ArchitectAgent()
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

                    # Update API call metrics
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
        return self.market_provider.get_stock_data(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'])

    def get_market_dashboard_data(self):
        """Get comprehensive market data for dashboard (restored from model4)"""
        return self.market_provider.get_market_dashboard_data()


# Initialize the manager
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

    # API Configuration Alert
    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API key not configured. Add GEMINI_API_KEY to Streamlit secrets for full functionality.")

    manager = get_manager()

    # Get current market data
    try:
        market_data = manager.get_current_market_data()
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        market_data = {}

    # Sidebar for system controls and market data
    with st.sidebar:
        # User info
        st.markdown(f"**👤 Logged in as:**")
        st.info(f"{st.session_state.user_email}")

        st.header("System Controls")

        if st.button("🔄 Refresh Market Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()

        if st.button("▶️ Execute All Pending Tasks", type="primary"):
            with st.spinner("Executing tasks with Gemini AI..."):
                completed = manager.execute_all_pending_tasks()
                st.success(f"Executed {len(completed)} tasks")

        # Display market data
        if market_data:
            display_market_data(market_data)

        st.header("System Status")
        status = manager.get_system_status()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Agents", status["active_agents"])
            st.metric("Pending Tasks", status["pending_tasks"])
        with col2:
            st.metric("Completed", status["completed_tasks"])
            st.metric("API Calls", status["total_api_calls"])

        st.write(f"**System Health:** {status['system_health']}")

    # Main dashboard tabs - RESTORED Market Data tab from model4
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Agent Overview", "Market Data", "Task Assignment", "Performance Metrics", "Task History"])

    with tab1:
        st.header("Agent Status Dashboard")

        # Create agent status cards
        cols = st.columns(3)
        for i, (key, agent) in enumerate(manager.agents.items()):
            with cols[i % 3]:
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

    # RESTORED Market Data Tab from model4
    with tab2:
        st.subheader("Stock & Crypto Overview")

        col_left, col_right = st.columns(2)

        with col_left:
            symbol = st.selectbox("Enter the Stock Symbol :",
                                  ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'])

            if st.button("Get Stock Data"):
                with st.spinner(f"Fetching data for {symbol}..."):
                    stock_data = manager.market_provider.get_stock_price(symbol.upper())

                    if "error" not in stock_data:
                        st.metric("Current Price", f"${stock_data['current_price']:.2f}")
                        daily_change = stock_data['current_price'] - stock_data['open']
                        daily_change_pct = (daily_change / stock_data['open']) * 100
                        st.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:+.2f}%")
                        st.metric("Volume", f"{stock_data['volume']:,}")

                        st.subheader(f"📊 Detailed Data for {symbol.upper()}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"Open: {stock_data['open']:.2f}")
                            st.write(f"High: {stock_data['high']:.2f}")
                        with col2:
                            st.write(f"Low: {stock_data['low']:.2f}")
                            st.write(f"P/E Ratio: {stock_data.get('pe_ratio', 0):.2f}")
                        with col3:
                            st.write(f"Market Cap: {stock_data.get('market_cap', 0):.2f}")

                        st.caption(f"⏱ Timestamp: {stock_data['timestamp']}")
                    else:
                        st.error(stock_data["error"])

        with col_right:
            st.subheader("💰 Crypto Prices")
            crypto_data = manager.market_provider.get_crypto_prices()
            for symbol, data in crypto_data.items():
                change_color = "🟢" if data["change_24h"] >= 0 else "🔴"
                st.metric(
                    label=symbol.replace("-USD", ""),
                    value=f"${data['current_price']:.2f}",
                    delta=f"{change_color} {data['change_24h']:+.2f}%"
                )

        # Market Overview Section
        st.subheader("Market Overview")
        market_overview = manager.market_provider.get_market_overview()

        if "error" not in market_overview:
            for index, data in market_overview.items():
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
                    f"{data['current']:.2f}",
                    f"{change_color} {data['change_percent']:.2f}%"
                )

    with tab3:
        st.header("Task Assignment")

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_agent = st.selectbox(
                "Select Agent:",
                options=list(manager.agents.keys()),
                format_func=lambda x: manager.agents[x].name
            )

            task_description = st.text_area("Task Description (mention stock symbols for data analysis):", height=100,
                                            placeholder="Enter a detailed task description...")

            priority = st.selectbox(
                "Priority:",
                options=[TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL],
                format_func=lambda x: x.name
            )

            if st.button("Assign Task"):
                if task_description:
                    result = manager.assign_task(selected_agent, task_description, priority)
                    st.success(result)
                else:
                    st.error("Please enter a task description")

        with col2:
            st.subheader("AI-Powered Quick Tasks")
            quick_tasks = {
                "Analyze AAPL Stock": "Provide comprehensive analysis of AAPL stock including technical indicators and market sentiment",
                "Market Risk Assessment": "Evaluate current market risk across all major indices and provide recommendations",
                "Crypto Market Analysis": "Analyze cryptocurrency market trends and identify opportunities",
                "Portfolio Optimization": "Suggest portfolio optimization strategies based on current market conditions",
                "Market Analysis": "Analyze current market conditions and identify trading opportunities",
                "Risk Assessment": "Evaluate current portfolio risk exposure and recommend adjustments",
                "Compliance Check": "Review recent trading activities for regulatory compliance",
                "Performance Report": "Generate comprehensive performance analysis and insights",
                "Strategy Review": "Review and optimize current trading strategies",
                "Data Quality Audit": "Assess data quality and identify potential issues"
            }

            for task_name, description in quick_tasks.items():
                if st.button(task_name, key=f"quick_{task_name}"):
                    # Auto-assign to appropriate agent
                    agent_mapping = {
                        "Analyze AAPL Stock": "quant_analyst",
                        "Market Risk Assessment": "risk_modeler",
                        "Crypto Market Analysis": "trading_strategist",
                        "Portfolio Optimization": "ml_engineer",
                        "Market Analysis": "quant_analyst",
                        "Risk Assessment": "risk_modeler",
                        "Compliance Check": "compliance_officer",
                        "Performance Report": "translator",
                        "Strategy Review": "trading_strategist",
                        "Data Quality Audit": "data_engineer"
                    }
                    result = manager.assign_task(
                        agent_mapping[task_name],
                        description,
                        TaskPriority.HIGH
                    )
                    st.success(result)

    with tab4:
        st.header("Performance Metrics")

        # System overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tasks", status["total_tasks"])
        with col2:
            success_rate = (status["completed_tasks"] / status["total_tasks"] * 100) if status["total_tasks"] > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            avg_response_time = np.mean([agent.metrics.avg_processing_time for agent in manager.agents.values()])
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        with col4:
            total_errors = sum(agent.metrics.errors_count for agent in manager.agents.values())
            st.metric("Total Errors", total_errors)

        # Agent performance chart
        st.subheader("Agent Performance Comparison")

        agent_data = []
        for key, agent in manager.agents.items():
            agent_data.append({
                "Agent": agent.name,
                "Tasks Completed": agent.metrics.tasks_completed,
                "Success Rate": agent.metrics.success_rate,
                "Avg Processing Time": agent.metrics.avg_processing_time,
                "Errors": agent.metrics.errors_count,
                "AI Enabled": "Yes" if agent.gemini_client.model else "No"
            })

        df = pd.DataFrame(agent_data)

        if not df.empty:
            fig = px.bar(df, x="Agent", y="Tasks Completed",
                         title="Tasks Completed by Agent",
                         color="Success Rate",
                         color_continuous_scale="Viridis")
            # fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            # Additional performance visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Success rate chart
                fig2 = px.bar(df, x="Agent", y="Success Rate",
                              title="Success Rate by Agent",
                              color="Success Rate",
                              color_continuous_scale="RdYlGn")
                # fig2.update_xaxis(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                # Processing time chart
                fig3 = px.bar(df, x="Agent", y="Avg Processing Time",
                              title="Average Processing Time by Agent",
                              color="Avg Processing Time",
                              color_continuous_scale="RdYlBu_r")
                # fig3.update_xaxis(tickangle=45)
                st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        st.header("Task History")

        if manager.task_history:
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status:",
                    ["Pending", "Completed", "Failed"],
                    default=["Pending", "Completed", "Failed"]
                )
            with col2:
                agent_filter = st.multiselect(
                    "Filter by Agent:",
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
                        "Description": task.description[:50] + "..." if len(
                            task.description) > 50 else task.description,
                        "Priority": task.priority.name,
                        "Status": task.status,
                        "Created": task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Completed": task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else "-",
                        "Result": task.result[:100] + "..." if task.result and len(
                            task.result) > 100 else task.result or "-",
                        "Has Market Data": "Yes" if task.market_data else "No"
                    })

            if task_data:
                df_tasks = pd.DataFrame(task_data)
                st.dataframe(df_tasks, use_container_width=True, height=400)

                # Show detailed task results
                if st.checkbox("Show Detailed Task Results"):
                    selected_task = st.selectbox("Select Task for Details:",
                                                 [task["Task ID"] for task in task_data])

                    task_detail = next((task for task in manager.task_history
                                        if task.id == selected_task), None)

                    if task_detail and task_detail.result:
                        st.subheader(f"Task Details: {selected_task}")
                        st.text_area("Full Result:", task_detail.result, height=200)

                        if task_detail.market_data:
                            st.subheader("Associated Market Data:")
                            # Handle both MarketData objects and dictionary formats
                            if isinstance(list(task_detail.market_data.values())[0], MarketData):
                                market_dict = {}
                                for symbol, data in task_detail.market_data.items():
                                    market_dict[symbol] = {
                                        "current_price": data.current_price,
                                        "change_percent": data.change_percent,
                                        "volume": data.volume,
                                        "timestamp": data.timestamp.isoformat()
                                    }
                                st.json(market_dict)
                            else:
                                st.json(task_detail.market_data)
            else:
                st.info("No tasks match the current filters")
        else:
            st.info("No task history available. Assign some tasks to get started!")

    # Footer with API status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        gemini_status = "🟢 Connected" if GEMINI_API_KEY else "🔴 Not Configured"
        st.write(f"**Gemini AI:** {gemini_status}")

    with col2:
        st.write(f"**Market Data:** 🟢 yfinance Active")

    with col3:
        st.write(f"**Last Updated:** {datetime.datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh option
    if st.checkbox("Auto-refresh (30s)", value=False):
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
