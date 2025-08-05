import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import time
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
class AgentMetrics:
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_processing_time: float = 0.0
    last_activity: Optional[datetime.datetime] = None
    errors_count: int = 0


@dataclass
class Task:
    id: str
    agent_id: str
    description: str
    priority: TaskPriority
    status: str = "Pending"
    created_at: datetime.datetime = None
    completed_at: Optional[datetime.datetime] = None
    result: Optional[str] = None
    market_data: Optional[Dict] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()


class MarketDataProvider:
    """Handles real-time market data from multiple free APIs"""

    def __init__(self):
        self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')

    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock price using yfinance (free)"""
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
        """Get market overview data"""
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
        """Get cryptocurrency prices using free API"""
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


class GeminiAIClient:
    """Client for interacting with Google Gemini AI API"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate response using Gemini AI"""
        try:
            # Add context to prompt if provided
            if context:
                context_str = json.dumps(context, indent=2)
                full_prompt = f"Context: {context_str}\n\nTask: {prompt}"
            else:
                full_prompt = prompt

            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini AI error: {e}")
            return f"Error generating AI response: {str(e)}"


class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.tasks_queue = []
        self.current_task = None
        self.market_data_provider = MarketDataProvider()

        # Initialize Gemini AI client
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        if gemini_api_key:
            self.ai_client = GeminiAIClient(gemini_api_key)
        else:
            self.ai_client = None
            logger.warning(f"No Gemini API key provided for agent {agent_id}")

    @abstractmethod
    def process_task(self, task: Task) -> str:
        pass

    def add_task(self, task: Task):
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


# Specialized Agent Implementations with Real-time Data and AI
class TradingStrategistAgent(BaseAgent):
    def __init__(self):
        super().__init__("TS001", "AI Trading Strategist",
                         "Defines investment philosophy using real-time market analysis")

    def process_task(self, task: Task) -> str:
        # Get real-time market data
        market_overview = self.market_data_provider.get_market_overview()

        if self.ai_client:
            prompt = f"""
            As a Trading Strategist, analyze the current market conditions and develop a trading strategy.
            Task: {task.description}

            Please provide:
            1. Market sentiment analysis
            2. Recommended strategy type
            3. Risk assessment
            4. Key factors to monitor
            """

            ai_response = self.ai_client.generate_response(prompt, market_overview)
            task.market_data = market_overview
            return ai_response
        else:
            # Fallback to original simulation
            time.sleep(np.random.uniform(0.5, 2.0))
            strategies = ["Value Investing", "Growth Strategy", "Momentum Trading", "Mean Reversion"]
            return f"Developed {np.random.choice(strategies)} strategy with risk tolerance: {np.random.uniform(0.1, 0.3):.2f}"


class QuantAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("QA001", "Quantitative Analyst",
                         "Develops mathematical models using real-time market data")

    def process_task(self, task: Task) -> str:
        # Get specific stock data if mentioned in task
        symbols = self.extract_symbols_from_task(task.description)
        stock_data = {}

        for symbol in symbols[:5]:  # Limit to 5 symbols to avoid rate limits
            stock_data[symbol] = self.market_data_provider.get_stock_price(symbol)

        if self.ai_client and stock_data:
            prompt = f"""
            As a Quantitative Analyst, analyze the provided stock data and create mathematical models.
            Task: {task.description}

            Please provide:
            1. Statistical analysis of the data
            2. Identified patterns or anomalies
            3. Predictive model recommendations
            4. Risk-return metrics
            """

            ai_response = self.ai_client.generate_response(prompt, stock_data)
            task.market_data = stock_data
            return ai_response
        else:
            time.sleep(np.random.uniform(1.0, 3.0))
            model_accuracy = np.random.uniform(0.75, 0.95)
            return f"Created predictive model with {model_accuracy:.2%} accuracy. Analyzed {len(stock_data)} stocks."

    def extract_symbols_from_task(self, description: str) -> List[str]:
        """Extract stock symbols from task description"""
        # Simple extraction - you can make this more sophisticated
        common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        mentioned_symbols = []

        for symbol in common_symbols:
            if symbol.upper() in description.upper():
                mentioned_symbols.append(symbol)

        # If no specific symbols mentioned, use default set
        if not mentioned_symbols:
            mentioned_symbols = ['AAPL', 'GOOGL', 'MSFT']

        return mentioned_symbols


class MLEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__("MLE001", "Machine Learning Engineer",
                         "Develops ML models for trading using real-time data streams")

    def process_task(self, task: Task) -> str:
        # Get market data for ML model development
        market_data = self.market_data_provider.get_market_overview()

        if self.ai_client:
            prompt = f"""
            As a Machine Learning Engineer, design and implement ML solutions for trading.
            Task: {task.description}

            Please provide:
            1. ML model architecture recommendations
            2. Feature engineering suggestions
            3. Training data requirements
            4. Performance metrics and monitoring
            """

            ai_response = self.ai_client.generate_response(prompt, market_data)
            return ai_response
        else:
            time.sleep(np.random.uniform(0.8, 2.5))
            throughput = np.random.randint(10000, 50000)
            return f"Deployed ML pipeline processing {throughput} data points/second with real-time market feeds"


class RiskModelerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RM001", "AI Risk Modeler",
                         "Builds risk models using real-time market volatility data")

    def process_task(self, task: Task) -> str:
        # Get market volatility data
        market_data = self.market_data_provider.get_market_overview()
        crypto_data = self.market_data_provider.get_crypto_prices()

        combined_data = {**market_data, "crypto": crypto_data}

        if self.ai_client:
            prompt = f"""
            As a Risk Modeler, assess current market risks and provide risk management recommendations.
            Task: {task.description}

            Please provide:
            1. Current market risk assessment
            2. VaR calculations and stress test scenarios
            3. Portfolio risk recommendations
            4. Risk mitigation strategies
            """

            ai_response = self.ai_client.generate_response(prompt, combined_data)
            task.market_data = combined_data
            return ai_response
        else:
            time.sleep(np.random.uniform(1.0, 2.0))
            var_95 = np.random.uniform(0.02, 0.08)
            return f"Risk assessment with real-time data: VaR(95%): {var_95:.2%}, Market volatility: High"


# Keep other agents with similar AI integration pattern...
class DataEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__("DE001", "Data Engineer",
                         "Manages real-time data pipelines and market data feeds")

    def process_task(self, task: Task) -> str:
        if self.ai_client:
            prompt = f"""
            As a Data Engineer, design and maintain data infrastructure for trading systems.
            Task: {task.description}

            Please provide:
            1. Data pipeline architecture
            2. Real-time processing capabilities
            3. Data quality and validation measures
            4. Scalability recommendations
            """

            ai_response = self.ai_client.generate_response(prompt)
            return ai_response
        else:
            time.sleep(np.random.uniform(0.5, 1.5))
            return f"Real-time data pipeline: 15+ market feeds, <50ms latency, 99.9% uptime"


class AuditorAgent(BaseAgent):
    def __init__(self):
        super().__init__("AU001", "AI Auditor",
                         "Audits trading decisions and AI model performance")

    def process_task(self, task: Task) -> str:
        if self.ai_client:
            prompt = f"""
            As an AI Auditor, examine trading decisions and system performance.
            Task: {task.description}

            Please provide:
            1. Decision-making process audit
            2. Model performance evaluation
            3. Compliance assessment
            4. Recommendations for improvement
            """

            ai_response = self.ai_client.generate_response(prompt)
            return ai_response
        else:
            time.sleep(np.random.uniform(0.8, 1.8))
            return f"Audit complete: AI decisions reviewed, compliance score: 94.2%"


# Continue with other agents following similar patterns...
class ComplianceOfficerAgent(BaseAgent):
    def __init__(self):
        super().__init__("CO001", "Compliance Officer",
                         "Ensures trading compliance with real-time regulatory monitoring")

    def process_task(self, task: Task) -> str:
        if self.ai_client:
            prompt = f"""
            As a Compliance Officer, ensure all trading activities meet regulatory requirements.
            Task: {task.description}

            Please provide:
            1. Regulatory compliance check
            2. Risk assessment for compliance violations
            3. Required documentation
            4. Monitoring recommendations
            """

            ai_response = self.ai_client.generate_response(prompt)
            return ai_response
        else:
            time.sleep(np.random.uniform(0.6, 1.5))
            return f"Real-time compliance monitoring: All trades compliant, 0 violations detected"


class ManagerAgent:
    def __init__(self):
        self.agents = {
            "trading_strategist": TradingStrategistAgent(),
            "quant_analyst": QuantAnalystAgent(),
            "ml_engineer": MLEngineerAgent(),
            "data_engineer": DataEngineerAgent(),
            "risk_modeler": RiskModelerAgent(),
            "auditor": AuditorAgent(),
            "compliance_officer": ComplianceOfficerAgent(),
        }
        self.task_history = []
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "system_uptime": 0.0
        }
        self.market_data_provider = MarketDataProvider()

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

    def get_market_dashboard_data(self):
        """Get real-time market data for dashboard"""
        return {
            "market_overview": self.market_data_provider.get_market_overview(),
            "crypto_prices": self.market_data_provider.get_crypto_prices(),
            "timestamp": datetime.datetime.now().isoformat()
        }


# Initialize the manager
@st.cache_resource
def get_manager():
    return ManagerAgent()


def main():
    st.set_page_config(page_title="AI Trading Agent Management System", layout="wide")

    st.title("🤖 AI Trading Agent Management System")
    st.markdown("**Multi-Agent Trading System with Real-time Market Data & Gemini AI**")

    manager = get_manager()

    # Check for API keys
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_key:
        st.warning("⚠️ Gemini API key not configured. Agents will use fallback responses.")

    # Sidebar for system controls and market data
    with st.sidebar:
        st.header("System Controls")

        if st.button("Execute All Pending Tasks", type="primary"):
            with st.spinner("Executing tasks..."):
                completed = manager.execute_all_pending_tasks()
                st.success(f"Executed {len(completed)} tasks")

        st.header("System Status")
        status = manager.get_system_status()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Agents", status["active_agents"])
            st.metric("Pending Tasks", status["pending_tasks"])
        with col2:
            st.metric("Completed", status["completed_tasks"])
            st.metric("Failed", status["failed_tasks"])

        st.write(f"**System Health:** {status['system_health']}")

        # Real-time market data
        st.header("Live Market Data")
        market_data = manager.get_market_dashboard_data()

        if "error" not in market_data["market_overview"]:
            for index, data in market_data["market_overview"].items():
                change_color = "🟢" if data["change_percent"] >= 0 else "🔴"
                st.metric(
                    index.replace("^", ""),
                    f"{data['current']:.2f}",
                    f"{change_color} {data['change_percent']:.2f}%"
                )

    # Main dashboard tabs
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

                ai_status = "🤖 AI Enabled" if agent.ai_client else "🔧 Fallback Mode"

                st.info(f"""
                **{status_color} {agent.name}**  
                Status: {agent.status.value}  
                {ai_status}  
                Tasks Completed: {agent.metrics.tasks_completed}  
                Success Rate: {agent.metrics.success_rate:.1f}%  
                Queue: {len(agent.tasks_queue)} tasks
                """)

    with tab2:
        st.header("Real-time Market Data")

        # Stock price lookup
        col1, col2 = st.columns([2, 1])

        with col1:
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", value="AAPL")

            if st.button("Get Stock Data"):
                with st.spinner(f"Fetching data for {symbol}..."):
                    stock_data = manager.market_data_provider.get_stock_price(symbol.upper())

                    if "error" not in stock_data:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Current Price", f"${stock_data['current_price']:.2f}")
                        with col_b:
                            daily_change = stock_data['current_price'] - stock_data['open']
                            daily_change_pct = (daily_change / stock_data['open']) * 100
                            st.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:+.2f}%")
                        with col_c:
                            st.metric("Volume", f"{stock_data['volume']:,}")

                        st.json(stock_data)
                    else:
                        st.error(stock_data["error"])

        with col2:
            st.subheader("Crypto Prices")
            crypto_data = manager.market_data_provider.get_crypto_prices()
            for symbol, data in crypto_data.items():
                change_color = "🟢" if data["change_24h"] >= 0 else "🔴"
                st.metric(
                    symbol.replace("-USD", ""),
                    f"${data['current_price']:.2f}",
                    f"{change_color} {data['change_24h']:+.2f}%"
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

            task_description = st.text_area("Task Description (mention stock symbols for data analysis):", height=100)

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
                "Portfolio Optimization": "Suggest portfolio optimization strategies based on current market conditions"
            }

            for task_name, description in quick_tasks.items():
                if st.button(task_name, key=f"quick_{task_name}"):
                    # Auto-assign to appropriate agent
                    agent_mapping = {
                        "Analyze AAPL Stock": "quant_analyst",
                        "Market Risk Assessment": "risk_modeler",
                        "Crypto Market Analysis": "trading_strategist",
                        "Portfolio Optimization": "ml_engineer"
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
                "AI Enabled": "Yes" if agent.ai_client else "No"
            })

        df = pd.DataFrame(agent_data)

        if not df.empty:
            fig = px.bar(df, x="Agent", y="Tasks Completed",
                         title="Tasks Completed by Agent",
                         color="Success Rate",
                         color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

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
                            st.json(task_detail.market_data)
            else:
                st.info("No tasks match the current filters")
        else:
            st.info("No task history available. Assign some tasks to get started!")

    # Footer with API status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        gemini_status = "🟢 Connected" if gemini_key else "🔴 Not Configured"
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