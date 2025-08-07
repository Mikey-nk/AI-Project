import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import time
import asyncio
import aiohttp
import requests
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

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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
    def __init__(self):
        self.alpha_vantage = None
        if ALPHA_VANTAGE_API_KEY:
            self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

    def get_stock_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch real-time stock data using multiple sources"""
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

    def get_market_overview(self) -> Dict[str, MarketData]:
        """Get data for major market indices and stocks"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        return self.get_stock_data(symbols)


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
                market_summary.append(
                    f"{symbol}: ${data.current_price:.2f} "
                    f"({data.change_percent:+.2f}%), Volume: {data.volume:,}"
                )
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
        return self.market_provider.get_market_overview()


# Initialize the manager
@st.cache_resource
def get_manager():
    return ManagerAgent()


def display_market_data(market_data: Dict[str, MarketData]):
    """Display current market data in the sidebar"""
    st.subheader("📈 Live Market Data")

    for symbol, data in market_data.items():
        delta_color = "normal" if data.change >= 0 else "inverse"
        st.metric(
            label=symbol,
            value=f"${data.current_price:.2f}",
            delta=f"{data.change_percent:+.2f}%",
            delta_color=delta_color
        )


def main():
    st.set_page_config(page_title="AI Trading Agent Management System", layout="wide")

    st.title("🤖 AI Trading Agent Management System")
    st.markdown("**Multi-Agent Trading System with Real-Time Market Data & Gemini AI**")

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

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Agent Overview", "Task Assignment", "Market Analysis", "Performance Metrics", "Task History"])

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

                st.info(f"""
                **{status_color} {agent.name}**  
                Status: {agent.status.value}  
                Tasks Completed: {agent.metrics.tasks_completed}  
                Success Rate: {agent.metrics.success_rate:.1f}%  
                API Calls: {agent.metrics.api_calls_made}  
                Queue: {len(agent.tasks_queue)} tasks
                """)

    with tab2:
        st.header("Task Assignment")

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_agent = st.selectbox(
                "Select Agent:",
                options=list(manager.agents.keys()),
                format_func=lambda x: manager.agents[x].name
            )

            task_description = st.text_area("Task Description:", height=100,
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
            st.subheader("Quick Tasks")
            quick_tasks = {
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
                        TaskPriority.MEDIUM
                    )
                    st.success(result)

    with tab3:
        st.header("Real-Time Market Analysis")

        if market_data:
            # Market overview
            col1, col2, col3, col4 = st.columns(4)

            prices = [data.current_price for data in market_data.values()]
            changes = [data.change_percent for data in market_data.values()]
            volumes = [data.volume for data in market_data.values()]

            with col1:
                st.metric("Avg Price", f"${np.mean(prices):.2f}")
            with col2:
                avg_change = np.mean(changes)
                st.metric("Market Trend", f"{avg_change:+.2f}%", delta_color="normal" if avg_change >= 0 else "inverse")
            with col3:
                st.metric("Total Volume", f"{sum(volumes):,}")
            with col4:
                st.metric("Symbols Tracked", len(market_data))

            # Create market visualization
            symbols = list(market_data.keys())
            prices = [market_data[symbol].current_price for symbol in symbols]
            changes = [market_data[symbol].change_percent for symbol in symbols]

            # Price chart
            fig1 = px.bar(x=symbols, y=prices, title="Current Stock Prices",
                          labels={'x': 'Symbol', 'y': 'Price ($)'})
            fig1.update_traces(marker_color='lightblue')
            st.plotly_chart(fig1, use_container_width=True)

            # Performance chart
            colors = ['green' if change >= 0 else 'red' for change in changes]
            fig2 = px.bar(x=symbols, y=changes, title="Daily Performance (%)",
                          labels={'x': 'Symbol', 'y': 'Change (%)'})
            fig2.update_traces(marker_color=colors)
            st.plotly_chart(fig2, use_container_width=True)

            # Detailed market data table
            st.subheader("Detailed Market Data")
            market_df = pd.DataFrame([
                {
                    "Symbol": symbol,
                    "Price": f"${data.current_price:.2f}",
                    "Change": f"{data.change:+.2f}",
                    "Change %": f"{data.change_percent:+.2f}%",
                    "Volume": f"{data.volume:,}",
                    "52W High": f"${data.high_52w:.2f}",
                    "52W Low": f"${data.low_52w:.2f}",
                    "Market Cap": f"${data.market_cap:,.0f}"
                }
                for symbol, data in market_data.items()
            ])
            st.dataframe(market_df, use_container_width=True)
        else:
            st.warning("No market data available. Please check your API configuration.")

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
            st.metric("Avg Response Time", f"{avg_response_time}")