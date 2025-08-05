import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from abc import ABC, abstractmethod
import logging

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

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()


class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.tasks_queue = []
        self.current_task = None

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


# Specialized Agent Implementations
class TradingStrategistAgent(BaseAgent):
    def __init__(self):
        super().__init__("TS001", "AI Trading Strategist",
                         "Defines investment philosophy, risk tolerance, and performance goals")

    def process_task(self, task: Task) -> str:
        # Simulate strategy development
        time.sleep(np.random.uniform(0.5, 2.0))
        strategies = ["Value Investing", "Growth Strategy", "Momentum Trading", "Mean Reversion"]
        return f"Developed {np.random.choice(strategies)} strategy with risk tolerance: {np.random.uniform(0.1, 0.3):.2f}"


class QuantAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("QA001", "Quantitative Analyst",
                         "Develops mathematical models to identify market inefficiencies")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(1.0, 3.0))
        model_accuracy = np.random.uniform(0.75, 0.95)
        return f"Created predictive model with {model_accuracy:.2%} accuracy. Identified {np.random.randint(3, 8)} market signals."


class MLEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__("MLE001", "Machine Learning Engineer",
                         "Transforms models into scalable, production-ready software")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.8, 2.5))
        throughput = np.random.randint(10000, 50000)
        return f"Deployed ML pipeline processing {throughput} data points/second with 99.{np.random.randint(5, 9)}% uptime"


class DataEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__("DE001", "Data Engineer",
                         "Designs and maintains data infrastructure and ETL processes")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.5, 1.5))
        data_sources = np.random.randint(5, 15)
        latency = np.random.uniform(10, 100)
        return f"ETL pipeline active: {data_sources} data sources, {latency:.1f}ms average latency"


class RiskModelerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RM001", "AI Risk Modeler",
                         "Builds models to assess potential risks and mitigate losses")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(1.0, 2.0))
        var_95 = np.random.uniform(0.02, 0.08)
        return f"Risk assessment complete: VaR(95%): {var_95:.2%}, Stress test scenarios: {np.random.randint(10, 20)}"


class AuditorAgent(BaseAgent):
    def __init__(self):
        super().__init__("AU001", "AI Auditor",
                         "Examines decision-making processes and ensures transparency")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.8, 1.8))
        compliance_score = np.random.uniform(0.85, 0.99)
        return f"Audit complete: Compliance score {compliance_score:.1%}, {np.random.randint(0, 3)} issues identified"


class TranslatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("TR001", "AI Translator",
                         "Bridges technical AI systems and non-technical stakeholders")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.3, 1.0))
        return f"Generated executive summary: {np.random.randint(5, 12)} key insights translated for stakeholders"


class ComplianceOfficerAgent(BaseAgent):
    def __init__(self):
        super().__init__("CO001", "Compliance Officer",
                         "Ensures adherence to financial regulations and ethical practices")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.6, 1.5))
        regulations = ["GDPR", "CCPA", "MiFID II", "Dodd-Frank"]
        return f"Compliance check: {np.random.choice(regulations)} compliant, {np.random.randint(0, 2)} violations detected"


class TrustAuthenticatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("TA001", "Trust Authenticator",
                         "Verifies factual accuracy and fairness of AI outputs")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.4, 1.2))
        accuracy_score = np.random.uniform(0.90, 0.99)
        return f"Trust verification: {accuracy_score:.1%} accuracy score, bias metrics within acceptable range"


class EthicistAgent(BaseAgent):
    def __init__(self):
        super().__init__("ET001", "AI Ethicist",
                         "Develops ethical guidelines and ensures fair, transparent decisions")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.7, 1.8))
        ethical_score = np.random.uniform(0.80, 0.95)
        return f"Ethical review: Score {ethical_score:.1%}, fairness index: {np.random.uniform(0.85, 0.98):.2%}"


class LegalGuarantorAgent(BaseAgent):
    def __init__(self):
        super().__init__("LG001", "Legal Guarantor",
                         "Takes legal responsibility for AI actions and decisions")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(0.8, 2.0))
        risk_level = np.random.choice(["Low", "Medium", "High"])
        return f"Legal review complete: Risk level {risk_level}, liability assessment documented"


class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__("AR001", "AI Agent Architect",
                         "Designs system architecture and integrations")

    def process_task(self, task: Task) -> str:
        time.sleep(np.random.uniform(1.0, 2.5))
        uptime = np.random.uniform(0.995, 0.999)
        return f"System architecture optimized: {uptime:.3%} uptime, {np.random.randint(15, 30)} integrations active"


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
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "system_uptime": 0.0
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


# Initialize the manager
@st.cache_resource
def get_manager():
    return ManagerAgent()


def main():
    st.set_page_config(page_title="AI Trading Agent Management System", layout="wide")

    st.title("🤖 AI Trading Agent Management System")
    st.markdown("**Multi-Agent Trading System Controller**")

    manager = get_manager()

    # Sidebar for system controls
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

    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Agent Overview", "Task Assignment", "Performance Metrics", "Task History"])

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

            task_description = st.text_area("Task Description:", height=100)

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
                "Market Analysis": "Analyze current market conditions",
                "Risk Assessment": "Evaluate portfolio risk exposure",
                "Compliance Check": "Review regulatory compliance",
                "Performance Report": "Generate performance summary"
            }

            for task_name, description in quick_tasks.items():
                if st.button(task_name, key=f"quick_{task_name}"):
                    # Auto-assign to appropriate agent
                    agent_mapping = {
                        "Market Analysis": "quant_analyst",
                        "Risk Assessment": "risk_modeler",
                        "Compliance Check": "compliance_officer",
                        "Performance Report": "translator"
                    }
                    result = manager.assign_task(
                        agent_mapping[task_name],
                        description,
                        TaskPriority.MEDIUM
                    )
                    st.success(result)

    with tab3:
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
                "Errors": agent.metrics.errors_count
            })

        df = pd.DataFrame(agent_data)

        if not df.empty:
            fig = px.bar(df, x="Agent", y="Tasks Completed",
                         title="Tasks Completed by Agent",
                         color="Success Rate",
                         color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
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
                            task.result) > 100 else task.result or "-"
                    })

            if task_data:
                df_tasks = pd.DataFrame(task_data)
                st.dataframe(df_tasks, use_container_width=True, height=400)
            else:
                st.info("No tasks match the current filters")
        else:
            st.info("No task history available. Assign some tasks to get started!")

    # Auto-refresh option
    if st.checkbox("Auto-refresh (30s)", value=False):
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()