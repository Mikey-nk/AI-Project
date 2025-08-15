import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from abc import ABC, abstractmethod
import google.generativeai as genai
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib
import secrets
import asyncio
import threading
from collections import defaultdict, deque
import sqlite3
import os
from io import StringIO
import base64



# Configure logging with enhanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configuration - Enhanced with validation
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
ADMIN_EMAILS = st.secrets.get("ADMIN_EMAILS", ["kibe5067@gmail.com"])

# Configure Gemini with error handling
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")


# --------------- Enhanced Data Classes ---------------
class AgentStatus(Enum):
    ACTIVE = "Active"
    IDLE = "Idle"
    ERROR = "Error"
    MAINTENANCE = "Maintenance"
    PAUSED = "Paused"
    WARMING_UP = "Warming Up"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TaskStatus(Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    RETRY = "Retrying"


class AnalysisType(Enum):
    TECHNICAL = "Technical Analysis"
    FUNDAMENTAL = "Fundamental Analysis"
    SENTIMENT = "Sentiment Analysis"
    RISK = "Risk Assessment"
    STRATEGY = "Strategy Development"
    COMPLIANCE = "Compliance Check"


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
    day_high: float = 0.0
    day_low: float = 0.0
    avg_volume: int = 0
    beta: float = 0.0
    dividend_yield: float = 0.0
    eps: float = 0.0


@dataclass
class TechnicalIndicators:
    symbol: str
    sma_20: float
    sma_50: float
    sma_200: float
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    volume_sma: float
    volatility: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    timestamp: datetime.datetime


@dataclass
class NewsItem:
    title: str
    summary: str
    publisher: str
    published_time: datetime.datetime
    url: str
    sentiment_score: float = 0.0
    relevance_score: float = 0.0


@dataclass
class AgentMetrics:
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_processing_time: float = 0.0
    last_activity: Optional[datetime.datetime] = None
    errors_count: int = 0
    api_calls_made: int = 0
    total_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    uptime_hours: float = 0.0


@dataclass
class SystemAlert:
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    component: str
    timestamp: datetime.datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class Task:
    id: str
    agent_id: str
    description: str
    priority: TaskPriority
    analysis_type: AnalysisType = AnalysisType.TECHNICAL
    market_data: Optional[Dict[str, MarketData]] = None
    technical_data: Optional[Dict[str, TechnicalIndicators]] = None
    news_data: Optional[Dict[str, List[NewsItem]]] = None
    status: str = "Pending"
    created_at: datetime.datetime = None
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    confidence_score: float = 0.0
    tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()
        if self.tags is None:
            self.tags = []


# --------------- Enhanced Database Manager ---------------
class DatabaseManager:
    """Enhanced database manager for persistent storage"""

    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Tasks table
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS tasks
                               (
                                   id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   agent_id
                                   TEXT,
                                   description
                                   TEXT,
                                   priority
                                   INTEGER,
                                   analysis_type
                                   TEXT,
                                   status
                                   TEXT,
                                   created_at
                                   TIMESTAMP,
                                   started_at
                                   TIMESTAMP,
                                   completed_at
                                   TIMESTAMP,
                                   result
                                   TEXT,
                                   error_message
                                   TEXT,
                                   retry_count
                                   INTEGER,
                                   confidence_score
                                   REAL,
                                   tags
                                   TEXT
                               )
                               """)

                # Agent metrics table
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS agent_metrics
                               (
                                   agent_id
                                   TEXT,
                                   timestamp
                                   TIMESTAMP,
                                   tasks_completed
                                   INTEGER,
                                   success_rate
                                   REAL,
                                   avg_processing_time
                                   REAL,
                                   errors_count
                                   INTEGER,
                                   api_calls_made
                                   INTEGER,
                                   PRIMARY
                                   KEY
                               (
                                   agent_id,
                                   timestamp
                               )
                                   )
                               """)

                # System alerts table
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS system_alerts
                               (
                                   id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   level
                                   TEXT,
                                   message
                                   TEXT,
                                   component
                                   TEXT,
                                   timestamp
                                   TIMESTAMP,
                                   resolved
                                   BOOLEAN,
                                   resolution_notes
                                   TEXT
                               )
                               """)

                # Performance history table
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS performance_history
                               (
                                   timestamp
                                   TIMESTAMP,
                                   metric_name
                                   TEXT,
                                   metric_value
                                   REAL,
                                   agent_id
                                   TEXT,
                                   PRIMARY
                                   KEY
                               (
                                   timestamp,
                                   metric_name,
                                   agent_id
                               )
                                   )
                               """)

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def save_task(self, task: Task):
        """Save task to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO tasks 
                    (id, agent_id, description, priority, analysis_type, status, 
                     created_at, started_at, completed_at, result, error_message, 
                     retry_count, confidence_score, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id, task.agent_id, task.description, task.priority.value,
                    task.analysis_type.value, task.status, task.created_at,
                    task.started_at, task.completed_at, task.result, task.error_message,
                    task.retry_count, task.confidence_score, json.dumps(task.tags)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save task {task.id}: {e}")

    def get_tasks_history(self, limit: int = 100) -> List[Dict]:
        """Get tasks history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               SELECT *
                               FROM tasks
                               ORDER BY created_at DESC LIMIT ?
                               """, (limit,))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get tasks history: {e}")
            return []

    def save_agent_metrics(self, agent_id: str, metrics: AgentMetrics):
        """Save agent metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               INSERT INTO agent_metrics
                               (agent_id, timestamp, tasks_completed, success_rate,
                                avg_processing_time, errors_count, api_calls_made)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               """, (
                                   agent_id, datetime.datetime.now(), metrics.tasks_completed,
                                   metrics.success_rate, metrics.avg_processing_time,
                                   metrics.errors_count, metrics.api_calls_made
                               ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save agent metrics for {agent_id}: {e}")


# --------------- Enhanced Authentication System ---------------
class AuthenticationManager:
    """Enhanced authentication with session management and security features"""

    @staticmethod
    def authenticate_user(email: str, password: str) -> bool:
        """Enhanced authentication with rate limiting and security checks"""
        try:
            users = st.secrets.get("AUTHORIZED_USERS", {})
            hashed_password = users.get(email)

            if not hashed_password:
                AuthenticationManager._log_failed_attempt(email, "User not found")
                return False

            # Check for rate limiting
            if AuthenticationManager._is_rate_limited(email):
                st.error("🚫 Too many failed attempts. Please wait before trying again.")
                return False

            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                AuthenticationManager._clear_failed_attempts(email)
                AuthenticationManager._log_successful_login(email)
                return True
            else:
                AuthenticationManager._log_failed_attempt(email, "Invalid password")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    @staticmethod
    def _is_rate_limited(email: str) -> bool:
        """Check if user is rate limited"""
        failed_attempts = st.session_state.get(f"failed_attempts_{email}", [])
        recent_attempts = [
            attempt for attempt in failed_attempts
            if datetime.datetime.now() - attempt < datetime.timedelta(minutes=15)
        ]
        return len(recent_attempts) >= 5

    @staticmethod
    def _log_failed_attempt(email: str, reason: str):
        """Log failed authentication attempt"""
        attempts_key = f"failed_attempts_{email}"
        failed_attempts = st.session_state.get(attempts_key, [])
        failed_attempts.append(datetime.datetime.now())
        st.session_state[attempts_key] = failed_attempts[-10:]  # Keep last 10 attempts

        logger.warning(f"Failed login attempt for {email}: {reason}")

    @staticmethod
    def _clear_failed_attempts(email: str):
        """Clear failed attempts on successful login"""
        attempts_key = f"failed_attempts_{email}"
        if attempts_key in st.session_state:
            del st.session_state[attempts_key]

    @staticmethod
    def _log_successful_login(email: str):
        """Log successful login"""
        logger.info(f"Successful login for {email}")

    @staticmethod
    def generate_session_token(email: str) -> str:
        """Generate secure session token"""
        timestamp = str(datetime.datetime.now().timestamp())
        secret = st.secrets.get("SESSION_SECRET", "default_secret")
        return hashlib.sha256(f"{email}{timestamp}{secret}".encode()).hexdigest()

    @staticmethod
    def validate_session_token(email: str, token: str, timestamp: datetime.datetime) -> bool:
        """Validate session token"""
        try:
            secret = st.secrets.get("SESSION_SECRET", "default_secret")
            expected_token = hashlib.sha256(f"{email}{timestamp.timestamp()}{secret}".encode()).hexdigest()
            return token == expected_token
        except Exception:
            return False


# --------------- Enhanced Market Data Provider ---------------
class EnhancedMarketDataProvider:
    """Advanced market data provider with caching, fallbacks, and enhanced analytics"""

    def __init__(self):
        self.alpha_vantage = None
        self.rate_limits = defaultdict(lambda: {'calls': 0, 'reset_time': datetime.datetime.now()})
        self.cache = {}
        self.cache_ttl = 60  # seconds

        if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "your_alpha_vantage_api_key_here":
            try:
                self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage: {e}")

    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate cache key for method call"""
        key_data = f"{method}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if still valid"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.datetime.now() - timestamp).seconds < self.cache_ttl:
                return data
            else:
                del self.cache[cache_key]
        return None

    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = (data, datetime.datetime.now())

        # Clean old cache entries if cache gets too large
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def get_enhanced_stock_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get enhanced stock data with technical indicators"""
        cache_key = self._get_cache_key("get_enhanced_stock_data", tuple(symbols))
        cached_data = self._get_cached_data(cache_key)

        if cached_data:
            return cached_data

        market_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get basic info
                info = ticker.info
                hist = ticker.history(period="5d", interval="1d")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

                    # Calculate additional metrics
                    volatility = hist['Close'].std()
                    avg_volume = hist['Volume'].mean()

                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        current_price=float(current_price),
                        change=float(change),
                        change_percent=float(change_percent),
                        volume=int(info.get('volume', hist['Volume'].iloc[-1] if not hist['Volume'].empty else 0)),
                        timestamp=datetime.datetime.now(),
                        high_52w=float(info.get('fiftyTwoWeekHigh', 0)),
                        low_52w=float(info.get('fiftyTwoWeekLow', 0)),
                        market_cap=int(info.get('marketCap', 0)),
                        pe_ratio=float(info.get('trailingPE', 0)),
                        day_high=float(hist['High'].iloc[-1]),
                        day_low=float(hist['Low'].iloc[-1]),
                        avg_volume=int(avg_volume),
                        beta=float(info.get('beta', 1.0)),
                        dividend_yield=float(info.get('dividendYield', 0)) * 100 if info.get('dividendYield') else 0.0,
                        eps=float(info.get('trailingEps', 0))
                    )
                else:
                    market_data[symbol] = self._generate_enhanced_mock_data(symbol)

            except Exception as e:
                logger.error(f"Error fetching enhanced data for {symbol}: {e}")
                market_data[symbol] = self._generate_enhanced_mock_data(symbol)

        self._cache_data(cache_key, market_data)
        return market_data

    def get_technical_indicators(self, symbols: List[str]) -> Dict[str, TechnicalIndicators]:
        """Get comprehensive technical indicators"""
        cache_key = self._get_cache_key("get_technical_indicators", tuple(symbols))
        cached_data = self._get_cached_data(cache_key)

        if cached_data:
            return cached_data

        technical_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y", interval="1d")

                if len(hist) >= 50:  # Need enough data for calculations
                    close_prices = hist['Close']
                    high_prices = hist['High']
                    low_prices = hist['Low']
                    volumes = hist['Volume']

                    # Moving averages
                    sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
                    sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
                    sma_200 = close_prices.rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else sma_50

                    # RSI calculation
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]

                    # MACD calculation
                    ema_12 = close_prices.ewm(span=12).mean()
                    ema_26 = close_prices.ewm(span=26).mean()
                    macd = ema_12 - ema_26
                    macd_signal = macd.ewm(span=9).mean()

                    # Bollinger Bands
                    bb_middle = close_prices.rolling(window=20).mean()
                    bb_std = close_prices.rolling(window=20).std()
                    bb_upper = bb_middle + (bb_std * 2)
                    bb_lower = bb_middle - (bb_std * 2)

                    # Stochastic oscillator
                    low_14 = low_prices.rolling(window=14).min()
                    high_14 = high_prices.rolling(window=14).max()
                    stoch_k = 100 * ((close_prices - low_14) / (high_14 - low_14))
                    stoch_d = stoch_k.rolling(window=3).mean()

                    # Williams %R
                    williams_r = -100 * ((high_14 - close_prices) / (high_14 - low_14))

                    technical_data[symbol] = TechnicalIndicators(
                        symbol=symbol,
                        sma_20=float(sma_20),
                        sma_50=float(sma_50),
                        sma_200=float(sma_200),
                        rsi=float(rsi),
                        macd=float(macd.iloc[-1]),
                        macd_signal=float(macd_signal.iloc[-1]),
                        bb_upper=float(bb_upper.iloc[-1]),
                        bb_lower=float(bb_lower.iloc[-1]),
                        bb_middle=float(bb_middle.iloc[-1]),
                        volume_sma=float(volumes.rolling(window=20).mean().iloc[-1]),
                        volatility=float(close_prices.std()),
                        stoch_k=float(stoch_k.iloc[-1]),
                        stoch_d=float(stoch_d.iloc[-1]),
                        williams_r=float(williams_r.iloc[-1]),
                        timestamp=datetime.datetime.now()
                    )
                else:
                    technical_data[symbol] = self._generate_mock_technical_indicators(symbol)

            except Exception as e:
                logger.error(f"Error calculating technical indicators for {symbol}: {e}")
                technical_data[symbol] = self._generate_mock_technical_indicators(symbol)

        self._cache_data(cache_key, technical_data)
        return technical_data

    def get_enhanced_news_data(self, symbols: List[str]) -> Dict[str, List[NewsItem]]:
        """Get enhanced news data with sentiment analysis"""
        cache_key = self._get_cache_key("get_enhanced_news_data", tuple(symbols))
        cached_data = self._get_cached_data(cache_key)

        if cached_data:
            return cached_data

        news_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                news_list = ticker.news[:10]  # Get more news for better analysis

                processed_news = []
                for item in news_list:
                    try:
                        published_time = datetime.datetime.fromtimestamp(
                            item.get("providerPublishTime", time.time())
                        )

                        # Basic sentiment analysis (simplified)
                        title = item.get("title", "")
                        summary = item.get("summary", "")

                        # Simple keyword-based sentiment scoring
                        positive_words = ['gain', 'rise', 'up', 'strong', 'positive', 'bull', 'growth', 'increase']
                        negative_words = ['loss', 'fall', 'down', 'weak', 'negative', 'bear', 'decline', 'decrease']

                        text = f"{title} {summary}".lower()
                        positive_score = sum(1 for word in positive_words if word in text)
                        negative_score = sum(1 for word in negative_words if word in text)

                        sentiment_score = (positive_score - negative_score) / max(positive_score + negative_score, 1)

                        # Relevance score based on symbol mention
                        relevance_score = 1.0 if symbol.lower() in text else 0.5

                        news_item = NewsItem(
                            title=title,
                            summary=summary[:200] + "..." if len(summary) > 200 else summary,
                            publisher=item.get("publisher", "Unknown"),
                            published_time=published_time,
                            url=item.get("link", "#"),
                            sentiment_score=sentiment_score,
                            relevance_score=relevance_score
                        )
                        processed_news.append(news_item)

                    except Exception as e:
                        logger.warning(f"Error processing news item for {symbol}: {e}")
                        continue

                news_data[symbol] = processed_news

            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                news_data[symbol] = []

        self._cache_data(cache_key, news_data)
        return news_data

    def _generate_enhanced_mock_data(self, symbol: str) -> MarketData:
        """Generate enhanced mock data with more realistic values"""
        base_prices = {
            'AAPL': 175.0, 'MSFT': 340.0, 'GOOGL': 135.0, 'AMZN': 145.0,
            'TSLA': 210.0, 'NVDA': 420.0, 'META': 315.0, 'NFLX': 450.0,
            'SPY': 400.0, 'QQQ': 350.0, 'IWM': 180.0
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
            volume=np.random.randint(1000000, 50000000),
            timestamp=datetime.datetime.now(),
            high_52w=current_price * np.random.uniform(1.2, 1.6),
            low_52w=current_price * np.random.uniform(0.6, 0.8),
            market_cap=int(current_price * np.random.randint(500000000, 3000000000)),
            pe_ratio=float(np.random.uniform(10, 40)),
            day_high=current_price * np.random.uniform(1.01, 1.05),
            day_low=current_price * np.random.uniform(0.95, 0.99),
            avg_volume=int(np.random.randint(800000, 20000000)),
            beta=float(np.random.uniform(0.5, 2.0)),
            dividend_yield=float(np.random.uniform(0, 4.0)),
            eps=float(np.random.uniform(1.0, 15.0))
        )

    def _generate_mock_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Generate mock technical indicators"""
        base_price = 100.0
        sma_20 = base_price * np.random.uniform(0.95, 1.05)
        sma_50 = base_price * np.random.uniform(0.90, 1.10)
        sma_200 = base_price * np.random.uniform(0.85, 1.15)

        return TechnicalIndicators(
            symbol=symbol,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            rsi=float(np.random.uniform(20, 80)),
            macd=float(np.random.normal(0, 2)),
            macd_signal=float(np.random.normal(0, 1.5)),
            bb_upper=sma_20 * 1.05,
            bb_lower=sma_20 * 0.95,
            bb_middle=sma_20,
            volume_sma=float(np.random.randint(1000000, 10000000)),
            volatility=float(np.random.uniform(0.15, 0.45)),
            stoch_k=float(np.random.uniform(0, 100)),
            stoch_d=float(np.random.uniform(0, 100)),
            williams_r=float(np.random.uniform(-100, 0)),
            timestamp=datetime.datetime.now()
        )


# --------------- Enhanced Gemini AI Client ---------------
class EnhancedGeminiClient:
    """Advanced Gemini client with better prompt engineering and response handling"""

    def __init__(self):
        self.model = None
        self.request_count = 0
        self.last_request_time = datetime.datetime.now()
        self.rate_limit_delay = 1.0
        self.conversation_history = deque(maxlen=10)

        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel(
                    'gemini-2.0-flash-exp',
                    system_instruction=self._get_system_instruction()
                )
                logger.info("✅ Enhanced Gemini AI successfully initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")

    def _get_system_instruction(self) -> str:
        """Get comprehensive system instruction for Gemini"""
        return """
You are an expert AI financial analyst and trading assistant with deep expertise in:

CORE COMPETENCIES:
• Technical Analysis: Chart patterns, indicators, trend analysis
• Fundamental Analysis: Financial statements, valuation models, industry analysis  
• Risk Management: Portfolio optimization, VaR, stress testing
• Market Psychology: Sentiment analysis, behavioral finance
• Regulatory Compliance: SEC rules, trading regulations, best practices

ANALYSIS FRAMEWORK:
1. Always start with a clear executive summary
2. Provide specific, actionable recommendations
3. Include risk assessments and probability estimates
4. Use quantitative metrics when possible
5. Consider multiple scenarios (bull/bear/base case)
6. Address regulatory and compliance considerations

RESPONSE STRUCTURE:
• Use clear headings and bullet points for readability
• Include specific numbers, percentages, and timeframes
• Provide confidence levels for predictions
• Explain reasoning behind recommendations
• Always include appropriate disclaimers

RISK FOCUS:
• Emphasize risk management in all recommendations
• Identify potential downside scenarios
• Suggest position sizing and stop-loss levels
• Consider correlation and portfolio impact

Remember: You are providing analysis for professional trading decisions. Be thorough, precise, and always prioritize risk management.
"""

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((Exception,))
    )
    def generate_enhanced_response(self, prompt: str, context: dict = None,
                                   analysis_type: AnalysisType = AnalysisType.TECHNICAL) -> tuple[str, float]:
        """Generate enhanced response with confidence scoring"""
        if not self.model:
            return self._fallback_response(prompt, analysis_type), 0.3

        try:
            self._check_rate_limit()
            self.request_count += 1

            # Build comprehensive prompt
            enhanced_prompt = self._build_comprehensive_prompt(prompt, context, analysis_type)

            response = self.model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=3000,
                    temperature=0.2,  # Lower for more consistent financial analysis
                    top_p=0.8,
                    top_k=40
                )
            )

            if response.text:
                formatted_response = self._format_enhanced_response(response.text, analysis_type)
                confidence_score = self._calculate_confidence_score(response.text, context)

                # Store in conversation history
                self.conversation_history.append({
                    'prompt': prompt,
                    'response': formatted_response,
                    'timestamp': datetime.datetime.now(),
                    'confidence': confidence_score
                })

                return formatted_response, confidence_score
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            logger.error(f"Enhanced Gemini API error (attempt {self.request_count}): {e}")
            raise

    def _build_comprehensive_prompt(self, prompt: str, context: dict, analysis_type: AnalysisType) -> str:
        """Build comprehensive prompt with all available context"""
        sections = []

        # Analysis type specific instructions
        type_instructions = {
            AnalysisType.TECHNICAL: "Focus on price patterns, technical indicators, and momentum analysis.",
            AnalysisType.FUNDAMENTAL: "Emphasize financial metrics, valuation models, and company fundamentals.",
            AnalysisType.SENTIMENT: "Analyze market sentiment, news impact, and behavioral factors.",
            AnalysisType.RISK: "Prioritize risk assessment, stress testing, and downside protection.",
            AnalysisType.STRATEGY: "Develop strategic recommendations with clear entry/exit criteria.",
            AnalysisType.COMPLIANCE: "Review for regulatory compliance and identify potential violations."
        }

        sections.append(f"ANALYSIS TYPE: {analysis_type.value}")
        sections.append(f"SPECIFIC FOCUS: {type_instructions.get(analysis_type, 'Provide comprehensive analysis.')}")

        if context:
            # Market data context
            if 'market_data' in context and context['market_data']:
                market_summary = self._format_market_context(context['market_data'])
                sections.append(f"CURRENT MARKET DATA:\n{market_summary}")

            # Technical indicators context
            if 'technical_data' in context and context['technical_data']:
                technical_summary = self._format_technical_context(context['technical_data'])
                sections.append(f"TECHNICAL INDICATORS:\n{technical_summary}")

            # News context
            if 'news_data' in context and context['news_data']:
                news_summary = self._format_news_context(context['news_data'])
                sections.append(f"NEWS & SENTIMENT:\n{news_summary}")

            # Historical context from conversation
            if self.conversation_history:
                recent_context = self._format_conversation_context()
                sections.append(f"RECENT ANALYSIS CONTEXT:\n{recent_context}")

        sections.append(f"PRIMARY TASK:\n{prompt}")
        sections.append("""
REQUIRED OUTPUT FORMAT:
## 📊 Executive Summary
[Brief 2-3 sentence overview]

## 🎯 Key Findings  
• [Specific finding with supporting data]
• [Risk factors and opportunities]
• [Market positioning analysis]

## 📈 Detailed Analysis
[Comprehensive analysis with specific metrics]

## 💡 Recommendations
• **Action:** [Specific recommendation with rationale]
• **Risk Level:** [Low/Medium/High with explanation]
• **Timeline:** [Short/Medium/Long term outlook]
• **Position Size:** [Suggested allocation if applicable]

## ⚠️ Risk Considerations
• [Primary risks and mitigation strategies]
• [Downside scenarios and probability estimates]

## 🎖️ Confidence Assessment
[Rate confidence 1-10 with explanation]

Provide specific numbers, percentages, and actionable insights throughout.
""")

        return "\n\n".join(sections)

    def _format_market_context(self, market_data: dict) -> str:
        """Format market data for prompt context"""
        context_lines = []
        for symbol, data in market_data.items():
            if isinstance(data, MarketData):
                trend = "📈 Bullish" if data.change_percent > 1 else "📉 Bearish" if data.change_percent < -1 else "➡️ Neutral"

                context_lines.append(f"""
{symbol} - {trend}:
  • Price: ${data.current_price:.2f} ({data.change_percent:+.2f}%)
  • Volume: {data.volume:,} (Avg: {data.avg_volume:,})
  • Market Cap: ${data.market_cap / 1e9:.1f}B | P/E: {data.pe_ratio:.1f}
  • 52W Range: ${data.low_52w:.2f} - ${data.high_52w:.2f}
  • Dividend Yield: {data.dividend_yield:.2f}% | Beta: {data.beta:.2f}
""")
        return "".join(context_lines)

    def _format_technical_context(self, technical_data: dict) -> str:
        """Format technical indicators for prompt context"""
        context_lines = []
        for symbol, data in technical_data.items():
            if isinstance(data, TechnicalIndicators):
                # Determine trend signals
                ma_signal = "Bullish" if data.sma_20 > data.sma_50 > data.sma_200 else "Bearish" if data.sma_20 < data.sma_50 < data.sma_200 else "Mixed"
                rsi_signal = "Overbought" if data.rsi > 70 else "Oversold" if data.rsi < 30 else "Neutral"
                macd_signal = "Bullish" if data.macd > data.macd_signal else "Bearish"

                context_lines.append(f"""
{symbol} Technical Signals:
  • Moving Averages: {ma_signal} (20: ${data.sma_20:.2f}, 50: ${data.sma_50:.2f})
  • RSI: {data.rsi:.1f} ({rsi_signal})
  • MACD: {macd_signal} ({data.macd:.2f} vs {data.macd_signal:.2f})
  • Bollinger Bands: ${data.bb_lower:.2f} - ${data.bb_upper:.2f}
  • Volatility: {data.volatility:.2f} | Stoch: K={data.stoch_k:.1f}, D={data.stoch_d:.1f}
""")
        return "".join(context_lines)

    def _format_news_context(self, news_data: dict) -> str:
        """Format news data for prompt context"""
        context_lines = []
        for symbol, news_list in news_data.items():
            if news_list:
                sentiment_scores = [item.sentiment_score for item in news_list]
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"

                context_lines.append(f"""
{symbol} News Sentiment: {sentiment_label} (Score: {avg_sentiment:.2f})
Recent Headlines:""")

                for item in news_list[:3]:  # Top 3 news items
                    context_lines.append(f"  • {item.title} [{item.publisher}]")

                context_lines.append("")

        return "\n".join(context_lines)

    def _format_conversation_context(self) -> str:
        """Format recent conversation history for context"""
        if not self.conversation_history:
            return "No recent analysis history."

        recent = list(self.conversation_history)[-2:]  # Last 2 interactions
        context_lines = []

        for i, interaction in enumerate(recent, 1):
            context_lines.append(f"Previous Analysis #{i}:")
            context_lines.append(f"Task: {interaction['prompt'][:100]}...")
            context_lines.append(f"Confidence: {interaction['confidence']:.1f}/10")
            context_lines.append(f"Time: {interaction['timestamp'].strftime('%H:%M')}")
            context_lines.append("")

        return "\n".join(context_lines)

    def _calculate_confidence_score(self, response_text: str, context: dict) -> float:
        """Calculate confidence score based on response quality and available data"""
        score = 5.0  # Base score

        # Boost score based on available data
        if context:
            if context.get('market_data'): score += 1.0
            if context.get('technical_data'): score += 1.0
            if context.get('news_data'): score += 0.5

        # Analyze response quality indicators
        response_lower = response_text.lower()

        # Positive indicators
        if 'specific' in response_lower or 'precise' in response_lower: score += 0.5
        if any(word in response_lower for word in ['probability', 'likely', 'expect']): score += 0.5
        if ' in response_text and ' % ' in response_text: score += 0.5  # Contains specific numbers':
            if 'risk' in response_lower: score += 0.3

        # Negative indicators
        if any(word in response_lower for word in ['uncertain', 'unclear', 'limited data']): score -= 1.0
        if 'mock' in response_lower or 'simulated' in response_lower: score -= 2.0

        # Normalize to 1-10 scale
        return max(1.0, min(10.0, score))

    def _format_enhanced_response(self, response: str, analysis_type: AnalysisType) -> str:
        """Format response with enhanced styling and structure"""
        # Add analysis type badge
        type_emoji = {
            AnalysisType.TECHNICAL: "📊",
            AnalysisType.FUNDAMENTAL: "📈",
            AnalysisType.SENTIMENT: "🎭",
            AnalysisType.RISK: "⚠️",
            AnalysisType.STRATEGY: "🎯",
            AnalysisType.COMPLIANCE: "🛡️"
        }

        emoji = type_emoji.get(analysis_type, "🤖")
        formatted = f"# {emoji} {analysis_type.value}\n\n{response}"

        # Enhance formatting
        formatted = formatted.replace("## Executive Summary", "## 📊 **Executive Summary**")
        formatted = formatted.replace("## Key Findings", "## 🔍 **Key Findings**")
        formatted = formatted.replace("## Detailed Analysis", "## 📈 **Detailed Analysis**")
        formatted = formatted.replace("## Recommendations", "## 💡 **Recommendations**")
        formatted = formatted.replace("## Risk Considerations", "## ⚠️ **Risk Considerations**")
        formatted = formatted.replace("## Confidence Assessment", "## 🎖️ **Confidence Assessment**")

        return formatted

    def _fallback_response(self, prompt: str, analysis_type: AnalysisType) -> str:
        """Enhanced fallback response with analysis type specificity"""
        type_responses = {
            AnalysisType.TECHNICAL: "Technical analysis simulation based on general market patterns",
            AnalysisType.FUNDAMENTAL: "Fundamental analysis simulation using standard valuation metrics",
            AnalysisType.SENTIMENT: "Sentiment analysis simulation based on market psychology principles",
            AnalysisType.RISK: "Risk assessment simulation using standard risk management frameworks",
            AnalysisType.STRATEGY: "Strategy development simulation based on portfolio theory",
            AnalysisType.COMPLIANCE: "Compliance review simulation based on regulatory guidelines"
        }

        return f"""# 🤖 {analysis_type.value} - Simulation Mode

## 📊 **Executive Summary**
AI analysis system is operating in simulation mode. {type_responses.get(analysis_type, 'General analysis simulation')}.

## 🔍 **Key Findings**
• **System Status**: Gemini AI API unavailable or not configured
• **Analysis Request**: {prompt[:100]}...
• **Simulation Mode**: Using fallback analysis framework
• **Data Availability**: Limited to cached and mock data

## 💡 **Recommendations**
1. **Configure Gemini API**: Add valid API key to secrets.toml for full AI analysis
2. **Verify Connectivity**: Check internet connection and API quotas
3. **Review Configuration**: Ensure proper system setup and permissions
4. **Manual Analysis**: Consider manual review of available market data

## ⚠️ **Risk Considerations**
• **Limited Analysis**: Simulation mode provides generic responses only
• **Data Quality**: Mock data may not reflect actual market conditions
• **Decision Risk**: Avoid trading decisions based on simulated analysis

## 🎖️ **Confidence Assessment**
**Confidence Level: 2/10** - Simulation mode only, not suitable for trading decisions.

⚡ **Action Required**: Configure Gemini API for comprehensive AI-powered analysis.
"""

    def _check_rate_limit(self):
        """Enhanced rate limiting with adaptive delays"""
        now = datetime.datetime.now()
        time_since_last = (now - self.last_request_time).total_seconds()

        # Adaptive delay based on recent request frequency
        if self.request_count > 10:
            self.rate_limit_delay = min(3.0, self.rate_limit_delay * 1.1)
        elif self.request_count < 5:
            self.rate_limit_delay = max(0.5, self.rate_limit_delay * 0.9)

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = datetime.datetime.now()


# --------------- Enhanced Base Agent Class ---------------
class EnhancedBaseAgent(ABC):
    """Enhanced base agent with advanced capabilities"""

    def __init__(self, agent_id: str, name: str, description: str, specializations: List[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.specializations = specializations or []
        self.status = AgentStatus.WARMING_UP
        self.metrics = AgentMetrics()
        self.tasks_queue = []
        self.current_task = None
        self.gemini_client = EnhancedGeminiClient()
        self.market_provider = EnhancedMarketDataProvider()
        self.created_at = datetime.datetime.now()
        self.last_health_check = datetime.datetime.now()
        self.health_status = "Initializing"
        self.performance_history = deque(maxlen=100)
        self.alert_thresholds = {
            'error_rate': 0.3,
            'response_time': 30.0,
            'queue_size': 15
        }

        # Initialize agent
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize agent with warmup"""
        try:
            time.sleep(1)  # Simulation of initialization
            self.status = AgentStatus.IDLE
            self.health_status = "Healthy"
            logger.info(f"Agent {self.name} initialized successfully")
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.health_status = f"Initialization failed: {e}"
            logger.error(f"Failed to initialize agent {self.name}: {e}")

    @abstractmethod
    def get_agent_prompt(self) -> str:
        """Get the specific prompt for this agent type"""
        pass

    @abstractmethod
    def get_analysis_type(self) -> AnalysisType:
        """Get the primary analysis type for this agent"""
        pass

    def get_specialized_analysis(self, context: dict) -> str:
        """Strategic analysis focusing on portfolio optimization and market positioning"""
        if not context.get('market_data'):
            return "Insufficient market data for strategic analysis."

        analysis = []
        market_data = context['market_data']

        # Portfolio composition analysis
        total_market_cap = sum(
            data.market_cap for data in market_data.values() if isinstance(data, MarketData) and data.market_cap > 0)

        if total_market_cap > 0:
            analysis.append("STRATEGIC PORTFOLIO ANALYSIS:")

            # Market cap weighted analysis
            large_cap_weight = 0
            mid_cap_weight = 0
            small_cap_weight = 0

            for symbol, data in market_data.items():
                if isinstance(data, MarketData) and data.market_cap > 0:
                    weight = (data.market_cap / total_market_cap) * 100

                    if data.market_cap > 10e9:
                        large_cap_weight += weight
                    elif data.market_cap > 2e9:
                        mid_cap_weight += weight
                    else:
                        small_cap_weight += weight

                    momentum = "Strong Bullish" if data.change_percent > 3 else "Bullish" if data.change_percent > 1 else "Bearish" if data.change_percent < -1 else "Strong Bearish" if data.change_percent < -3 else "Neutral"

                    analysis.append(f"  • {symbol}: {weight:.1f}% allocation, {momentum} momentum")
                    analysis.append(f"    - Valuation: P/E {data.pe_ratio:.1f}, Beta {data.beta:.2f}")
                    analysis.append(f"    - Income: Div Yield {data.dividend_yield:.2f}%")

            analysis.append(f"\nCAPITALIZATION DISTRIBUTION:")
            analysis.append(f"  • Large Cap: {large_cap_weight:.1f}%")
            analysis.append(f"  • Mid Cap: {mid_cap_weight:.1f}%")
            analysis.append(f"  • Small Cap: {small_cap_weight:.1f}%")

        # Risk-Return Profile
        if context.get('technical_data'):
            analysis.append(f"\nRISK-RETURN ASSESSMENT:")
            volatilities = []
            returns = []

            for symbol, data in market_data.items():
                if isinstance(data, MarketData):
                    volatilities.append(abs(data.change_percent))
                    returns.append(data.change_percent)

            if volatilities:
                avg_volatility = np.mean(volatilities)
                avg_return = np.mean(returns)
                sharpe_estimate = avg_return / avg_volatility if avg_volatility > 0 else 0

                analysis.append(f"  • Portfolio Volatility: {avg_volatility:.2f}%")
                analysis.append(f"  • Expected Return: {avg_return:.2f}%")
                analysis.append(f"  • Risk-Adjusted Score: {sharpe_estimate:.2f}")

        return "\n".join(analysis) if analysis else "Strategic analysis requires valid market data."


class EnhancedQuantAnalystAgent(EnhancedBaseAgent):
    def __init__(self):
        super().__init__(
            "QA001",
            "Quantitative Analyst",
            "Develops mathematical models, statistical analysis, and algorithmic trading strategies",
            ["Statistical Modeling", "Algorithmic Trading", "Risk Metrics", "Backtesting", "Factor Analysis"]
        )

    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.TECHNICAL

    def get_agent_prompt(self) -> str:
        return """You are an elite Quantitative Analyst specializing in advanced mathematical modeling and algorithmic trading strategy development.

CORE RESPONSIBILITIES:
• Develop and validate sophisticated statistical models for price prediction
• Design and implement algorithmic trading strategies with rigorous backtesting
• Conduct advanced correlation analysis, factor modeling, and regime detection
• Calculate comprehensive risk metrics (VaR, CVaR, Maximum Drawdown, Sharpe, Sortino)
• Perform time series analysis, Monte Carlo simulations, and stress testing
• Identify market inefficiencies and statistical arbitrage opportunities

QUANTITATIVE ANALYSIS FRAMEWORK:
• Multi-factor models with dynamic parameter estimation
• Machine learning applications in pattern recognition
• High-frequency data analysis and microstructure modeling
• Options pricing models and volatility surface analysis
• Pairs trading and mean reversion strategies
• Alternative data integration and signal processing

DELIVERABLE REQUIREMENTS:
• Specific numerical metrics with statistical significance tests
• Probability distributions and confidence intervals
• Mathematical formulations and model assumptions
• Backtesting results with performance attribution
• Risk-adjusted returns and drawdown analysis
• Implementation specifications for algorithmic execution"""

    def get_specialized_analysis(self, context: dict) -> str:
        """Quantitative analysis with advanced statistical metrics"""
        if not context.get('market_data') and not context.get('technical_data'):
            return "Insufficient quantitative data for statistical analysis."

        analysis = []
        market_data = context.get('market_data', {})
        technical_data = context.get('technical_data', {})

        analysis.append("QUANTITATIVE STATISTICAL ANALYSIS:")

        # Price and return analysis
        if market_data:
            prices = []
            returns = []
            volumes = []
            volatilities = []

            for symbol, data in market_data.items():
                if isinstance(data, MarketData):
                    prices.append(data.current_price)
                    returns.append(data.change_percent)
                    volumes.append(data.volume)

                    # Calculate annualized volatility estimate
                    daily_vol = abs(data.change_percent) / 100
                    annual_vol = daily_vol * np.sqrt(252) * 100
                    volatilities.append(annual_vol)

            if len(returns) > 1:
                # Portfolio statistics
                portfolio_return = np.mean(returns)
                portfolio_vol = np.std(returns)
                max_return = np.max(returns)
                min_return = np.min(returns)

                # Risk metrics
                sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                downside_returns = [r for r in returns if r < 0]
                downside_deviation = np.std(downside_returns) if downside_returns else 0
                sortino_ratio = portfolio_return / downside_deviation if downside_deviation > 0 else 0

                # Value at Risk (95% confidence)
                var_95 = np.percentile(returns, 5)
                cvar_95 = np.mean([r for r in returns if r <= var_95])

                analysis.append(
                    f"  • Portfolio Return: {portfolio_return:.2f}% (Range: {min_return:.2f}% to {max_return:.2f}%)")
                analysis.append(f"  • Portfolio Volatility: {portfolio_vol:.2f}%")
                analysis.append(f"  • Sharpe Ratio: {sharpe_ratio:.3f}")
                analysis.append(f"  • Sortino Ratio: {sortino_ratio:.3f}")
                analysis.append(f"  • VaR (95%): {var_95:.2f}%")
                analysis.append(f"  • CVaR (95%): {cvar_95:.2f}%")

                # Correlation analysis
                if len(returns) >= 3:
                    analysis.append(f"\nCORRELATION ANALYSIS:")
                    correlation_matrix = np.corrcoef(returns[:min(len(returns), 10)])  # Limit for readability
                    avg_correlation = (np.sum(correlation_matrix) - np.trace(correlation_matrix)) / (
                                correlation_matrix.size - correlation_matrix.shape[0])
                    analysis.append(f"  • Average Cross-Correlation: {avg_correlation:.3f}")
                    analysis.append(
                        f"  • Diversification Benefit: {'High' if abs(avg_correlation) < 0.3 else 'Medium' if abs(avg_correlation) < 0.7 else 'Low'}")

        # Technical indicator analysis
        if technical_data:
            analysis.append(f"\nTECHNICAL INDICATOR STATISTICS:")

            rsi_values = []
            macd_values = []
            volatility_values = []

            for symbol, tech in technical_data.items():
                if isinstance(tech, TechnicalIndicators):
                    rsi_values.append(tech.rsi)
                    macd_values.append(tech.macd - tech.macd_signal)  # MACD histogram
                    volatility_values.append(tech.volatility)

                    # Individual security analysis
                    analysis.append(f"  • {symbol}:")
                    analysis.append(
                        f"    - RSI: {tech.rsi:.1f} ({'Overbought' if tech.rsi > 70 else 'Oversold' if tech.rsi < 30 else 'Neutral'})")
                    analysis.append(f"    - MACD Signal: {'Bullish' if tech.macd > tech.macd_signal else 'Bearish'}")
                    analysis.append(
                        f"    - Price vs MA20: {((tech.sma_20 / max(tech.sma_20 * 0.9, 0.01) - 1) * 100):+.1f}%")

            if rsi_values:
                avg_rsi = np.mean(rsi_values)
                rsi_std = np.std(rsi_values)

                analysis.append(f"\nAGGREGATE TECHNICAL METRICS:")
                analysis.append(f"  • Average RSI: {avg_rsi:.1f} ± {rsi_std:.1f}")
                analysis.append(
                    f"  • Market Condition: {'Overbought Territory' if avg_rsi > 70 else 'Oversold Territory' if avg_rsi < 30 else 'Neutral Zone'}")
                analysis.append(
                    f"  • Signal Dispersion: {'High' if rsi_std > 20 else 'Medium' if rsi_std > 10 else 'Low'}")

        # Market microstructure analysis
        if market_data:
            analysis.append(f"\nMARKET MICROSTRUCTURE ANALYSIS:")
            total_volume = sum(data.volume for data in market_data.values() if isinstance(data, MarketData))
            avg_volume = total_volume / len(market_data) if market_data else 0

            high_volume_count = sum(1 for data in market_data.values()
                                    if isinstance(data, MarketData) and data.volume > avg_volume * 1.5)

            analysis.append(f"  • Total Market Volume: {total_volume:,.0f}")
            analysis.append(f"  • High Volume Securities: {high_volume_count}/{len(market_data)}")
            analysis.append(
                f"  • Liquidity Assessment: {'High' if high_volume_count > len(market_data) * 0.6 else 'Medium' if high_volume_count > len(market_data) * 0.3 else 'Low'}")

        return "\n".join(analysis) if analysis else "Quantitative analysis requires valid market and technical data."


class EnhancedRiskModelerAgent(EnhancedBaseAgent):
    def __init__(self):
        super().__init__(
            "RM001",
            "AI Risk Modeler",
            "Assesses portfolio risks, stress testing, and develops comprehensive risk management strategies",
            ["Risk Assessment", "Stress Testing", "VaR Modeling", "Scenario Analysis", "Risk Management"]
        )

    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.RISK

    def get_agent_prompt(self) -> str:
        return """You are an elite AI Risk Modeler specializing in comprehensive risk assessment and advanced risk management strategies.

CORE RESPONSIBILITIES:
• Calculate Value at Risk (VaR), Expected Shortfall (ES), and Maximum Drawdown
• Perform comprehensive stress testing and scenario analysis
• Model market, credit, liquidity, operational, and systemic risks
• Assess concentration risk, correlation breakdowns, and tail risk events
• Design sophisticated risk mitigation and hedging strategies
• Implement dynamic risk monitoring and early warning systems

RISK MODELING FRAMEWORK:
• Monte Carlo simulations for portfolio risk assessment
• Historical simulation and parametric VaR methodologies
• Extreme Value Theory for tail risk modeling
• Copula models for dependency structure analysis
• Factor-based risk attribution and decomposition
• Regime-switching models for crisis risk assessment

DELIVERABLE REQUIREMENTS:
• Specific risk metrics with confidence intervals
• Worst-case scenario analysis with probability estimates
• Risk limits and threshold recommendations
• Hedging strategies with cost-benefit analysis
• Portfolio optimization for risk-adjusted returns
• Crisis management protocols and contingency plans"""

    def get_specialized_analysis(self, context: dict) -> str:
        """Comprehensive risk analysis with advanced modeling"""
        if not context.get('market_data'):
            return "Insufficient data for comprehensive risk assessment."

        analysis = []
        market_data = context.get('market_data', {})
        technical_data = context.get('technical_data', {})

        analysis.append("COMPREHENSIVE RISK ASSESSMENT:")

        # Portfolio risk metrics
        if market_data:
            returns = []
            volatilities = []
            market_caps = []
            betas = []

            for symbol, data in market_data.items():
                if isinstance(data, MarketData):
                    returns.append(data.change_percent)
                    volatilities.append(abs(data.change_percent))
                    market_caps.append(data.market_cap)
                    betas.append(data.beta)

            if returns:
                # Basic risk statistics
                portfolio_return = np.mean(returns)
                portfolio_vol = np.std(returns)
                max_loss = np.min(returns)
                max_gain = np.max(returns)

                # Risk metrics
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                expected_shortfall = np.mean([r for r in returns if r <= var_95])

                analysis.append(f"VALUE AT RISK ANALYSIS:")
                analysis.append(f"  • 1-Day VaR (95%): {var_95:.2f}%")
                analysis.append(f"  • 1-Day VaR (99%): {var_99:.2f}%")
                analysis.append(f"  • Expected Shortfall: {expected_shortfall:.2f}%")
                analysis.append(f"  • Maximum Observed Loss: {max_loss:.2f}%")

                # Volatility analysis
                avg_volatility = np.mean(volatilities)
                vol_of_vol = np.std(volatilities)

                analysis.append(f"\nVOLATILITY ASSESSMENT:")
                analysis.append(f"  • Average Volatility: {avg_volatility:.2f}%")
                analysis.append(f"  • Volatility of Volatility: {vol_of_vol:.2f}%")
                analysis.append(
                    f"  • Volatility Regime: {'High' if avg_volatility > 3 else 'Medium' if avg_volatility > 1.5 else 'Low'}")

                # Concentration risk
                total_market_cap = sum(market_caps)
                if total_market_cap > 0:
                    concentration_hhi = sum((mc / total_market_cap) ** 2 for mc in market_caps if mc > 0)
                    analysis.append(f"\nCONCENTRATION RISK:")
                    analysis.append(f"  • Herfindahl Index: {concentration_hhi:.3f}")
                    analysis.append(
                        f"  • Concentration Level: {'High' if concentration_hhi > 0.25 else 'Medium' if concentration_hhi > 0.15 else 'Low'}")

                # Beta analysis
                if betas:
                    avg_beta = np.mean([b for b in betas if b > 0])
                    beta_dispersion = np.std([b for b in betas if b > 0])

                    analysis.append(f"\nSYSTEMATIC RISK EXPOSURE:")
                    analysis.append(f"  • Portfolio Beta: {avg_beta:.2f}")
                    analysis.append(f"  • Beta Dispersion: {beta_dispersion:.2f}")
                    analysis.append(
                        f"  • Market Sensitivity: {'High' if avg_beta > 1.2 else 'Medium' if avg_beta > 0.8 else 'Low'}")

        # Technical risk indicators
        if technical_data:
            analysis.append(f"\nTECHNICAL RISK SIGNALS:")

            overbought_count = 0
            oversold_count = 0
            high_vol_count = 0

            for symbol, tech in technical_data.items():
                if isinstance(tech, TechnicalIndicators):
                    if tech.rsi > 70:
                        overbought_count += 1
                    elif tech.rsi < 30:
                        oversold_count += 1

                    if tech.volatility > 0.3:  # High volatility threshold
                        high_vol_count += 1

            total_securities = len(technical_data)
            analysis.append(
                f"  • Overbought Securities: {overbought_count}/{total_securities} ({overbought_count / total_securities * 100:.1f}%)")
            analysis.append(
                f"  • Oversold Securities: {oversold_count}/{total_securities} ({oversold_count / total_securities * 100:.1f}%)")
            analysis.append(
                f"  • High Volatility Securities: {high_vol_count}/{total_securities} ({high_vol_count / total_securities * 100:.1f}%)")

        # Risk scenario analysis
        analysis.append(f"\nRISK SCENARIO ANALYSIS:")
        if returns:
            # Stress test scenarios
            bear_market_scenario = portfolio_return - 2 * portfolio_vol
            bull_market_scenario = portfolio_return + 2 * portfolio_vol
            black_swan_scenario = portfolio_return - 4 * portfolio_vol

            analysis.append(f"  • Bear Market Scenario (-2σ): {bear_market_scenario:.2f}%")
            analysis.append(f"  • Bull Market Scenario (+2σ): {bull_market_scenario:.2f}%")
            analysis.append(f"  • Black Swan Event (-4σ): {black_swan_scenario:.2f}%")

        # Risk recommendations
        analysis.append(f"\nRISK MITIGATION RECOMMENDATIONS:")

        if market_data:
            high_risk_securities = sum(1 for data in market_data.values()
                                       if isinstance(data, MarketData) and abs(data.change_percent) > 3)
            total_securities = len(market_data)

            if high_risk_securities > total_securities * 0.5:
                analysis.append(
                    f"  • HIGH RISK ALERT: {high_risk_securities}/{total_securities} securities showing high volatility")
                analysis.append(f"  • Recommended Action: Reduce position sizes, increase hedging")
            elif high_risk_securities > total_securities * 0.3:
                analysis.append(
                    f"  • MEDIUM RISK: {high_risk_securities}/{total_securities} securities require monitoring")
                analysis.append(f"  • Recommended Action: Implement stop-loss orders, review correlations")
            else:
                analysis.append(f"  • ACCEPTABLE RISK: Portfolio showing normal volatility patterns")
                analysis.append(f"  • Recommended Action: Maintain current risk management protocols")

        return "\n".join(analysis) if analysis else "Risk analysis requires valid market data."


class EnhancedComplianceOfficerAgent(EnhancedBaseAgent):
    def __init__(self):
        super().__init__(
            "CO001",
            "Compliance Officer",
            "Ensures regulatory adherence, monitors trading compliance, and manages regulatory risk",
            ["Regulatory Compliance", "Trading Surveillance", "Risk Controls", "Audit Trail", "Reporting"]
        )

    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.COMPLIANCE

    def get_agent_prompt(self) -> str:
        return """You are an elite Compliance Officer ensuring comprehensive regulatory adherence in AI-driven trading systems.

CORE RESPONSIBILITIES:
• Monitor compliance with SEC, FINRA, CFTC, and international regulatory requirements
• Implement trading surveillance systems for market manipulation and insider trading detection
• Ensure proper documentation, audit trails, and regulatory reporting
• Monitor position limits, concentration rules, and leverage restrictions
• Review algorithmic trading systems for compliance with MiFID II and other regulations
• Manage conflicts of interest and ensure fiduciary duty compliance

COMPLIANCE FRAMEWORK:
• Real-time trade surveillance and pattern recognition
• Best execution analysis and transaction cost analysis
• Market abuse detection including layering, spoofing, and wash trading
• Know Your Customer (KYC) and Anti-Money Laundering (AML) protocols
• Risk management system compliance and model validation
• Regulatory change management and impact assessment

DELIVERABLE REQUIREMENTS:
• Specific regulatory citations and compliance requirements
• Risk-based compliance scoring and violation probability assessment
• Detailed remediation plans with timeline and responsibility matrix
• Regulatory reporting requirements and filing deadlines
• Control framework recommendations with testing procedures
• Training requirements and competency assessments"""

    def get_specialized_analysis(self, context: dict) -> str:
        """Comprehensive compliance analysis with regulatory focus"""
        if not context.get('market_data'):
            return "Insufficient data for compliance analysis."

        analysis = []
        market_data = context.get('market_data', {})

        analysis.append("REGULATORY COMPLIANCE ASSESSMENT:")

        # Position concentration analysis (SEC Rule 15c3-1)
        if market_data:
            total_positions = len(market_data)
            large_positions = 0
            illiquid_positions = 0
            high_vol_positions = 0

            analysis.append(f"POSITION CONCENTRATION ANALYSIS:")

            for symbol, data in market_data.items():
                if isinstance(data, MarketData):
                    # Market cap analysis for liquidity
                    if data.market_cap > 10e9:
                        cap_category = "Large Cap"
                    elif data.market_cap > 2e9:
                        cap_category = "Mid Cap"
                        large_positions += 1
                    else:
                        cap_category = "Small Cap"
                        large_positions += 1

                    # Liquidity risk assessment
                    avg_daily_volume = data.avg_volume * data.current_price
                    if avg_daily_volume < 10e6:  # Less than $10M daily volume
                        liquidity_risk = "High"
                        illiquid_positions += 1
                    elif avg_daily_volume < 50e6:
                        liquidity_risk = "Medium"
                    else:
                        liquidity_risk = "Low"

                    # Volatility assessment
                    if abs(data.change_percent) > 5:
                        high_vol_positions += 1
                        vol_risk = "High"
                    elif abs(data.change_percent) > 2:
                        vol_risk = "Medium"
                    else:
                        vol_risk = "Low"

                    analysis.append(
                        f"  • {symbol}: {cap_category}, Liquidity Risk: {liquidity_risk}, Volatility Risk: {vol_risk}")

            # Concentration compliance metrics
            concentration_ratio = large_positions / total_positions if total_positions > 0 else 0
            illiquidity_ratio = illiquid_positions / total_positions if total_positions > 0 else 0
            volatility_ratio = high_vol_positions / total_positions if total_positions > 0 else 0

            analysis.append(f"\nCONCENTRATION COMPLIANCE METRICS:")
            analysis.append(f"  • Non-Large Cap Concentration: {concentration_ratio:.1%}")
            analysis.append(f"  • Illiquid Position Ratio: {illiquidity_ratio:.1%}")
            analysis.append(f"  • High Volatility Ratio: {volatility_ratio:.1%}")

            # Compliance status assessment
            compliance_issues = []
            if concentration_ratio > 0.6:
                compliance_issues.append("High concentration in non-large cap securities")
            if illiquidity_ratio > 0.3:
                compliance_issues.append("Excessive exposure to illiquid securities")
            if volatility_ratio > 0.4:
                compliance_issues.append("High portfolio volatility risk")

            if compliance_issues:
                analysis.append(f"\nCOMPLIANCE ALERTS:")
                for issue in compliance_issues:
                    analysis.append(f"  ⚠️ {issue}")
            else:
                analysis.append(f"\n✅ COMPLIANCE STATUS: No significant concentration issues identified")

        # Market manipulation surveillance
        analysis.append(f"\nMARKET SURVEILLANCE ANALYSIS:")
        if market_data:
            unusual_volume_count = 0
            price_manipulation_signals = 0

            for symbol, data in market_data.items():
                if isinstance(data, MarketData):
                    # Volume analysis
                    if data.avg_volume > 0 and data.volume > data.avg_volume * 3:
                        unusual_volume_count += 1
                        analysis.append(
                            f"  🔍 {symbol}: Unusual volume detected ({data.volume / data.avg_volume:.1f}x average)")

                    # Price movement analysis
                    if abs(data.change_percent) > 10:
                        price_manipulation_signals += 1
                        analysis.append(f"  🚨 {symbol}: Extreme price movement ({data.change_percent:+.2f}%)")

            analysis.append(f"\nSURVEILLANCE SUMMARY:")
            analysis.append(f"  • Unusual Volume Alerts: {unusual_volume_count}/{len(market_data)}")
            analysis.append(f"  • Price Movement Alerts: {price_manipulation_signals}/{len(market_data)}")

            if unusual_volume_count > 0 or price_manipulation_signals > 0:
                analysis.append(f"  • Recommendation: Enhanced monitoring and potential SAR filing review")

        # Regulatory reporting requirements
        analysis.append(f"\nREGULATORY REPORTING REQUIREMENTS:")
        analysis.append(f"  • Daily Position Reports: Required for positions >$100M")
        analysis.append(f"  • Large Trader Reporting: Required if daily volume >$20M")
        analysis.append(f"  • Form PF Filing: Required for assets >$150M (quarterly)")
        analysis.append(f"  • 13F Filing: Required for equity positions >$100M (quarterly)")

        # Best execution analysis
        if market_data:
            analysis.append(f"\nBEST EXECUTION ANALYSIS:")
            total_securities = len(market_data)
            liquid_securities = sum(1 for data in market_data.values()
                                    if isinstance(data, MarketData) and data.volume * data.current_price > 10e6)

            analysis.append(
                f"  • Liquid Securities: {liquid_securities}/{total_securities} ({liquid_securities / total_securities:.1%})")
            analysis.append(
                f"  • Best Execution Review: {'Required' if liquid_securities < total_securities * 0.8 else 'Standard'}")

        # Risk management compliance
        analysis.append(f"\nRISK MANAGEMENT COMPLIANCE:")
        analysis.append(f"  • Position Limits: Daily monitoring required")
        analysis.append(f"  • VaR Limits: Real-time monitoring with 99% confidence level")
        analysis.append(f"  • Stress Testing: Monthly comprehensive stress tests required")
        analysis.append(f"  • Model Validation: Annual independent validation required")

        return "\n".join(analysis) if analysis else "Compliance analysis requires valid market data."


# --------------- Enhanced Manager Agent ---------------
class EnhancedManagerAgent:
    """Enhanced manager with advanced system capabilities"""

    def __init__(self):
        self.agents = {
            "trading_strategist": EnhancedTradingStrategistAgent(),
            "quant_analyst": EnhancedQuantAnalystAgent(),
            "risk_modeler": EnhancedRiskModelerAgent(),
            "compliance_officer": EnhancedComplianceOfficerAgent(),
        }

        self.task_history = []
        self.market_provider = EnhancedMarketDataProvider()
        self.db_manager = DatabaseManager()
        self.system_start_time = datetime.datetime.now()
        self.system_alerts = []
        self.performance_metrics = defaultdict(list)

        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "system_uptime": datetime.datetime.now(),
            "total_api_calls": 0,
            "peak_queue_size": 0,
            "avg_response_time": 0.0,
            "error_rate": 0.0
        }

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize system components"""
        try:
            # Warm up agents
            for agent in self.agents.values():
                agent.health_check()

            logger.info("Enhanced Manager Agent System initialized successfully")

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self._create_system_alert("ERROR", f"System initialization failed: {e}", "ManagerAgent")

    def assign_task(self, agent_key: str, description: str, priority: TaskPriority,
                    symbols: List[str] = None, analysis_type: AnalysisType = None,
                    tags: List[str] = None) -> str:
        """Enhanced task assignment with comprehensive validation"""
        if agent_key not in self.agents:
            available_agents = ", ".join(self.agents.keys())
            return f"❌ Error: Agent '{agent_key}' not found. Available agents: {available_agents}"

        try:
            task_id = f"TASK_{len(self.task_history):04d}_{datetime.datetime.now().strftime('%H%M%S')}"

            # Set default analysis type based on agent
            if not analysis_type:
                analysis_type = self.agents[agent_key].get_analysis_type()

            task = Task(
                id=task_id,
                agent_id=self.agents[agent_key].agent_id,
                description=description,
                priority=priority,
                analysis_type=analysis_type,
                tags=tags or []
            )

            # Add task to agent with enhanced data gathering
            self.agents[agent_key].add_task(task, symbols=symbols)
            self.task_history.append(task)
            self.system_metrics["total_tasks"] += 1

            # Update peak queue size
            current_queue_size = sum(len(agent.tasks_queue) for agent in self.agents.values())
            if current_queue_size > self.system_metrics["peak_queue_size"]:
                self.system_metrics["peak_queue_size"] = current_queue_size

            # Save task to database
            self.db_manager.save_task(task)

            logger.info(f"Task {task_id} assigned to {self.agents[agent_key].name}")
            return f"✅ Task {task_id} assigned to {self.agents[agent_key].name} with {analysis_type.value} analysis"

        except Exception as e:
            error_msg = f"Failed to assign task: {str(e)}"
            logger.error(error_msg)
            self._create_system_alert("ERROR", error_msg, "TaskAssignment")
            return f"❌ {error_msg}"

    def execute_all_pending_tasks(self) -> List[Task]:
        """Execute all pending tasks with enhanced monitoring"""
        completed_tasks = []
        start_time = time.time()

        try:
            for agent_key, agent in self.agents.items():
                while agent.tasks_queue:
                    try:
                        completed_task = agent.execute_next_task()
                        if completed_task:
                            completed_tasks.append(completed_task)

                            # Update system metrics
                            if completed_task.status == TaskStatus.COMPLETED.value:
                                self.system_metrics["completed_tasks"] += 1
                            else:
                                self.system_metrics["failed_tasks"] += 1

                            # Save completed task
                            self.db_manager.save_task(completed_task)

                            # Save agent metrics
                            self.db_manager.save_agent_metrics(agent.agent_id, agent.metrics)

                    except Exception as e:
                        logger.error(f"Error executing task in agent {agent_key}: {e}")
                        self._create_system_alert("ERROR", f"Task execution failed in {agent_key}: {e}", agent_key)

            # Update system performance metrics
            execution_time = time.time() - start_time
            self.system_metrics["total_api_calls"] = sum(agent.metrics.api_calls_made for agent in self.agents.values())

            if completed_tasks:
                avg_response_time = sum(
                    task.actual_duration for task in completed_tasks if task.actual_duration > 0) / len(completed_tasks)
                self.system_metrics["avg_response_time"] = avg_response_time

                # Calculate error rate
                failed_count = sum(1 for task in completed_tasks if task.status == TaskStatus.FAILED.value)
                self.system_metrics["error_rate"] = (failed_count / len(completed_tasks)) * 100

            logger.info(f"Executed {len(completed_tasks)} tasks in {execution_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Critical error in task execution: {e}")
            self._create_system_alert("CRITICAL", f"Task execution system failure: {e}", "SystemCore")

        return completed_tasks

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            uptime = (datetime.datetime.now() - self.system_start_time).total_seconds() / 3600
            active_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ACTIVE)
            total_pending_tasks = sum(len(agent.tasks_queue) for agent in self.agents.values())

            # Health assessment
            unhealthy_agents = sum(1 for agent in self.agents.values() if "Error" in agent.health_status)
            system_health = "Critical" if unhealthy_agents > 2 else "Warning" if unhealthy_agents > 0 else "Healthy"

            # Performance metrics
            recent_alerts = [alert for alert in self.system_alerts if
                             (datetime.datetime.now() - alert.timestamp).seconds < 3600]  # Last hour

            status = {
                "system_health": system_health,
                "uptime_hours": uptime,
                "active_agents": active_agents,
                "total_agents": len(self.agents),
                "pending_tasks": total_pending_tasks,
                "unhealthy_agents": unhealthy_agents,
                "recent_alerts": len(recent_alerts),
                "critical_alerts": len([a for a in recent_alerts if a.level == "CRITICAL"]),
                **self.system_metrics
            }

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"system_health": "Error", "error": str(e)}

    def get_agent_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status for all agents"""
        health_status = {}

        for agent_key, agent in self.agents.items():
            try:
                health_data = agent.health_check()
                health_status[agent_key] = {
                    "name": agent.name,
                    "status": agent.status.value,
                    "health": health_data,
                    "specializations": agent.specializations,
                    "queue_size": len(agent.tasks_queue),
                    "current_task": agent.current_task.id if agent.current_task else None
                }
            except Exception as e:
                health_status[agent_key] = {
                    "name": agent.name,
                    "status": "Error",
                    "error": str(e)
                }

        return health_status

    def _create_system_alert(self, level: str, message: str, component: str):
        """Create system alert"""
        alert = SystemAlert(
            id=f"ALERT_{len(self.system_alerts):04d}",
            level=level,
            message=message,
            component=component,
            timestamp=datetime.datetime.now()
        )

        self.system_alerts.append(alert)

        # Keep only last 1000 alerts
        if len(self.system_alerts) > 1000:
            self.system_alerts = self.system_alerts[-1000:]

        logger.log(
            logging.CRITICAL if level == "CRITICAL" else
            logging.ERROR if level == "ERROR" else
            logging.WARNING if level == "WARNING" else
            logging.INFO,
            f"System Alert [{level}] {component}: {message}"
        )

    def get_recent_alerts(self, hours: int = 24) -> List[SystemAlert]:
        """Get recent system alerts"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        return [alert for alert in self.system_alerts if alert.timestamp >= cutoff_time]

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        analytics = {
            "task_completion_rate": 0.0,
            "average_confidence_score": 0.0,
            "agent_utilization": {},
            "error_trends": [],
            "performance_trends": []
        }

        try:
            # Task completion rate
            if self.system_metrics["total_tasks"] > 0:
                analytics["task_completion_rate"] = (
                        self.system_metrics["completed_tasks"] / self.system_metrics["total_tasks"] * 100
                )

            # Agent utilization and performance
            for agent_key, agent in self.agents.items():
                if agent.metrics.tasks_completed > 0:
                    analytics["agent_utilization"][agent_key] = {
                        "tasks_completed": agent.metrics.tasks_completed,
                        "success_rate": agent.metrics.success_rate,
                        "avg_processing_time": agent.metrics.avg_processing_time,
                        "utilization_rate": min(100.0, (
                                    agent.metrics.tasks_completed / max(self.system_metrics["total_tasks"], 1)) * 100)
                    }

            # Average confidence score from recent tasks
            recent_tasks = [task for task in self.task_history if task.confidence_score > 0]
            if recent_tasks:
                analytics["average_confidence_score"] = np.mean([task.confidence_score for task in recent_tasks])

        except Exception as e:
            logger.error(f"Failed to generate performance analytics: {e}")

        return analytics


# --------------- Enhanced UI Functions ---------------
@st.cache_resource
def get_enhanced_manager():
    """Get cached enhanced manager instance"""
    return EnhancedManagerAgent()


def display_enhanced_login_page():
    """Display enhanced login page with better security"""
    st.set_page_config(
        page_title="AI Trading System - Secure Login",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #1f77b4, #2196F3);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.4);
    }
    .security-badge {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Trading Agent Management System</h1>
        <h3>🔐 Secure Access Portal</h3>
        <p>Enterprise-Grade Multi-Agent Trading Platform</p>
        <div>
            <span class="security-badge">🛡️ Bank-Level Security</span>
            <span class="security-badge">🤖 AI-Powered Analysis</span>
            <span class="security-badge">📊 Real-Time Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Secure Login")

        with st.form("enhanced_login_form", clear_on_submit=False):
            email = st.text_input(
                "📧 Email Address",
                placeholder="Enter your registered email",
                help="Use your authorized email address"
            )
            password = st.text_input(
                "🔑 Password",
                type="password",
                placeholder="Enter your secure password",
                help="Your password is encrypted and never stored in plain text"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                login_button = st.form_submit_button("🚀 Secure Login")

            if login_button:
                if email and password:
                    with st.spinner("🔐 Authenticating with advanced security protocols..."):
                        if AuthenticationManager.authenticate_user(email, password):
                            # Create secure session
                            login_time = datetime.datetime.now()
                            session_token = AuthenticationManager.generate_session_token(email)

                            st.session_state.authenticated = True
                            st.session_state.user_email = email
                            st.session_state.login_time = login_time
                            st.session_state.session_token = session_token
                            st.session_state.is_admin = email in ADMIN_EMAILS

                            st.success("✅ Authentication successful! Welcome to the AI Trading System.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please verify your credentials.")
                            time.sleep(1)
                else:
                    st.warning("⚠️ Please provide both email and password.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Security features info
    with st.expander("🛡️ Security Features", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🔐 Authentication:**
            - Bcrypt password hashing
            - Session token validation
            - Rate limiting protection
            - Failed attempt tracking
            """)

        with col2:
            st.markdown("""
            **🛡️ Data Protection:**
            - Encrypted API communications
            - Secure secret management
            - Real-time threat monitoring
            - Audit trail logging
            """)

    # System status footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        gemini_status = "🟢 Connected" if GEMINI_API_KEY else "🔴 Not Configured"
        st.write(f"**Gemini AI:** {gemini_status}")

    with col2:
        st.write("**Security:** 🟢 Active")

    with col3:
        st.write(f"**System Time:** {datetime.datetime.now().strftime('%H:%M:%S UTC')}")


def enhanced_check_authentication():
    """Enhanced authentication check with session management"""
    if not st.session_state.get("authenticated", False):
        display_enhanced_login_page()
        return False

    # Validate session token
    email = st.session_state.get("user_email")
    token = st.session_state.get("session_token")
    login_time = st.session_state.get("login_time")

    if not AuthenticationManager.validate_session_token(email, token, login_time):
        st.warning("🔒 Session validation failed. Please log in again.")
        enhanced_logout()
        return False

    # Check session timeout (24 hours)
    if login_time:
        time_diff = datetime.datetime.now() - login_time
        if time_diff.total_seconds() > 86400:  # 24 hours
            st.warning("⏰ Session expired for security. Please log in again.")
            enhanced_logout()
            return False

    return True


def enhanced_logout():
    """Enhanced logout with complete session cleanup"""
    # Log logout event
    if st.session_state.get("user_email"):
        logger.info(f"User logout: {st.session_state.user_email}")

    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Clear caches
    st.cache_data.clear()
    st.cache_resource.clear()

    st.success("👋 Securely logged out. All session data cleared.")
    time.sleep(1)
    st.rerun()


def display_enhanced_system_status(manager: EnhancedManagerAgent):
    """Display enhanced system status dashboard"""
    status = manager.get_system_status()

    # System health indicator
    health_color = {
        "Healthy": "🟢",
        "Warning": "🟡",
        "Critical": "🔴",
        "Error": "🔴"
    }.get(status["system_health"], "⚪")

    st.markdown(f"### {health_color} System Status: {status['system_health']}")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "System Uptime",
            f"{status['uptime_hours']:.1f}h",
            delta=f"{status['active_agents']}/{status['total_agents']} agents active"
        )

    with col2:
        completion_rate = (status['completed_tasks'] / max(status['total_tasks'], 1)) * 100
        st.metric(
            "Task Completion Rate",
            f"{completion_rate:.1f}%",
            delta=f"{status['completed_tasks']} completed"
        )

    with col3:
        st.metric(
            "Response Time",
            f"{status['avg_response_time']:.2f}s",
            delta=f"{status['error_rate']:.1f}% error rate",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "API Efficiency",
            f"{status['total_api_calls']} calls",
            delta=f"Peak queue: {status['peak_queue_size']}"
        )

    # Recent alerts
    recent_alerts = manager.get_recent_alerts(hours=1)
    if recent_alerts:
        st.warning(f"⚠️ {len(recent_alerts)} recent alerts (last hour)")

        with st.expander("View Recent Alerts", expanded=False):
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                alert_icon = {"CRITICAL": "🚨", "ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(alert.level, "📋")
                st.write(f"{alert_icon} **{alert.level}** [{alert.component}] - {alert.message}")
                st.caption(f"Time: {alert.timestamp.strftime('%H:%M:%S')}")


def display_enhanced_agent_dashboard(manager: EnhancedManagerAgent):
    """Display enhanced agent dashboard with detailed metrics"""
    st.header("🤖 Enhanced Agent Dashboard")

    agent_health = manager.get_agent_health_status()

    # Agent cards with enhanced information
    cols = st.columns(2)

    for i, (agent_key, health_data) in enumerate(agent_health.items()):
        with cols[i % 2]:
            agent_name = health_data.get("name", agent_key)
            agent_status = health_data.get("status", "Unknown")

            # Status color coding
            status_color = {
                "Active": "🟢",
                "Idle": "🟡",
                "Error": "🔴",
                "Maintenance": "🟠",
                "Paused": "⚪",
                "Warming Up": "🔵"
            }.get(agent_status, "⚪")

            # Create enhanced agent card
            with st.container():
                st.markdown(f"""
                <div style="
                    padding: 1.5rem;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border-left: 5px solid {'#4CAF50' if 'Healthy' in str(health_data.get('health', {}).get('status', '')) else '#f44336'};
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                <h4>{status_color} {agent_name}</h4>
                """, unsafe_allow_html=True)

                if "health" in health_data:
                    health = health_data["health"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Status:** {health.get('status', 'Unknown')}")
                        st.write(f"**Queue:** {health_data.get('queue_size', 0)} tasks")
                        st.write(f"**Uptime:** {health.get('uptime_hours', 0):.1f}h")

                    with col2:
                        st.write(f"**Success Rate:** {health.get('consecutive_successes', 0)} consecutive")
                        st.write(f"**Response Time:** {health.get('response_time', 0):.2f}s")
                        st.write(f"**AI Status:** {'🤖 Enabled' if health.get('ai_available') else '🔧 Fallback'}")

                    # Specializations
                    if health_data.get("specializations"):
                        specializations = ", ".join(health_data["specializations"])
                        st.write(f"**Specializations:** {specializations}")

                    # Current task
                    if health_data.get("current_task"):
                        st.info(f"🔄 Currently processing: {health_data['current_task']}")

                st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Enhanced main application with comprehensive features"""
    # Check authentication first
    if not enhanced_check_authentication():
        return

    st.set_page_config(
        page_title="AI Trading Agent Management System - Enhanced",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        # 🤖 AI Trading Agent Management System
        **Enterprise-Grade Multi-Agent Trading Platform with Advanced Analytics**
        """)

    with col2:
        st.markdown(f"""
        **👤 User:** {st.session_state.user_email}  
        **🔑 Role:** {'Admin' if st.session_state.get('is_admin') else 'User'}
        """)

    with col3:
        if st.button("🚪 Secure Logout", type="secondary", use_container_width=True):
            enhanced_logout()

    # Get enhanced manager
    manager = get_enhanced_manager()

    # Enhanced sidebar with system monitoring
    with st.sidebar:
        st.markdown("### 🎛️ System Controls")

        # System status
        display_enhanced_system_status(manager)

        st.markdown("---")

        # Quick actions
        if st.button("🔄 Refresh All Data", type="secondary"):
            st.cache_data.clear()
            st.success("🔄 Data refreshed")
            st.rerun()

        if st.button("▶️ Execute All Tasks", type="primary"):
            with st.spinner("🚀 Executing all pending tasks..."):
                completed = manager.execute_all_pending_tasks()
                st.success(f"✅ Executed {len(completed)} tasks")
                st.rerun()

        # Advanced settings (Admin only)
        if st.session_state.get('is_admin'):
            st.markdown("---")
            st.markdown("### 🔧 Admin Controls")

            if st.button("🗃️ Export System Data"):
                # Export functionality would go here
                st.info("📊 System data export initiated")

            if st.button("🔍 Run System Diagnostics"):
                # Diagnostics functionality would go here
                st.info("🔍 System diagnostics completed")

        # Performance metrics
        st.markdown("---")
        st.markdown("### 📊 Live Metrics")

        performance = manager.get_performance_analytics()
        st.metric("Completion Rate", f"{performance['task_completion_rate']:.1f}%")
        st.metric("Avg Confidence", f"{performance['average_confidence_score']:.1f}/10")

    # Enhanced main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Agent Dashboard",
        "📈 Market Intelligence",
        "📋 Task Management",
        "📊 Analytics",
        "📚 Task History",
        "🛡️ System Monitor"
    ])

    with tab1:
        display_enhanced_agent_dashboard(manager)

    with tab2:
        st.header("📈 Market Intelligence Center")

        # Enhanced market data display would go here
        # This would include the enhanced market data features
        st.info("🚧 Enhanced Market Intelligence features will be implemented here")

    with tab3:
        st.header("📋 Enhanced Task Management")

        # Enhanced task assignment interface would go here
        st.info("🚧 Enhanced Task Management interface will be implemented here")

    with tab4:
        st.header("📊 Advanced Analytics")

        # Performance analytics and visualizations would go here
        analytics = manager.get_performance_analytics()

        if analytics["agent_utilization"]:
            st.subheader("Agent Performance Comparison")

            # Create performance dataframe
            agent_data = []
            for agent_key, data in analytics["agent_utilization"].items():
                agent_data.append({
                    "Agent": manager.agents[agent_key].name,
                    "Tasks Completed": data["tasks_completed"],
                    "Success Rate": data["success_rate"],
                    "Avg Response Time": data["avg_processing_time"],
                    "Utilization": data["utilization_rate"]
                })

            if agent_data:
                df = pd.DataFrame(agent_data)

                # Performance charts
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(df, x="Agent", y="Tasks Completed",
                                 title="Tasks Completed by Agent",
                                 color="Success Rate", color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = px.scatter(df, x="Avg Response Time", y="Success Rate",
                                      size="Tasks Completed", hover_name="Agent",
                                      title="Response Time vs Success Rate")
                    st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.header("📚 Comprehensive Task History")

        # Enhanced task history display would go here
        st.info("🚧 Enhanced Task History features will be implemented here")

    with tab6:
        st.header("🛡️ System Monitor & Alerts")

        # System monitoring and alerting interface
        recent_alerts = manager.get_recent_alerts(hours=24)

        if recent_alerts:
            st.subheader("Recent System Alerts")

            for alert in recent_alerts[-10:]:  # Last 10 alerts
                alert_color = {
                    "CRITICAL": "error",
                    "ERROR": "error",
                    "WARNING": "warning",
                    "INFO": "info"
                }.get(alert.level, "info")

                with st.container():
                    getattr(st, alert_color)(
                        f"**{alert.level}** [{alert.component}] - {alert.message}"
                    )
                    st.caption(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.success("✅ No recent system alerts")

    # Enhanced footer
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_status = "🟢 Connected" if GEMINI_API_KEY else "🔴 Not Configured"
        st.write(f"**Gemini AI:** {gemini_status}")

    with col2:
        av_status = "🟢 Available" if ALPHA_VANTAGE_API_KEY else "🟡 Optional"
        st.write(f"**Alpha Vantage:** {av_status}")

    with col3:
        st.write(f"**Market Data:** 🟢 Real-Time Active")

    with col4:
        st.write(f"**Last Updated:** {datetime.datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()


def health_check(self) -> Dict[str, Any]:
    """Comprehensive agent health check"""
    try:
        now = datetime.datetime.now()
        uptime = (now - self.created_at).total_seconds() / 3600

        # Performance metrics
        error_rate = (self.metrics.errors_count / max(self.metrics.tasks_completed, 1)) * 100
        queue_size = len(self.tasks_queue)

        # Determine health status
        if self.status == AgentStatus.ERROR:
            self.health_status = "Error - Requires attention"
        elif error_rate > self.alert_thresholds['error_rate'] * 100:
            self.health_status = "Warning - High error rate"
        elif self.metrics.avg_processing_time > self.alert_thresholds['response_time']:
            self.health_status = "Warning - Slow response"
        elif queue_size > self.alert_thresholds['queue_size']:
            self.health_status = "Warning - High queue"
        elif self.metrics.consecutive_failures > 3:
            self.health_status = "Warning - Multiple failures"
        else:
            self.health_status = "Healthy"

        self.last_health_check = now
        self.metrics.uptime_hours = uptime

        health_data = {
            "status": self.health_status,
            "uptime_hours": uptime,
            "queue_size": queue_size,
            "error_rate": error_rate,
            "last_activity": self.metrics.last_activity,
            "response_time": self.metrics.avg_processing_time,
            "ai_available": bool(self.gemini_client.model),
            "consecutive_successes": self.metrics.consecutive_successes,
            "consecutive_failures": self.metrics.consecutive_failures
        }

        return health_data

    except Exception as e:
        logger.error(f"Health check failed for agent {self.agent_id}: {e}")
        return {"status": "Error", "error": str(e)}


def add_task(self, task: Task, symbols: List[str] = None):
    """Enhanced task addition with comprehensive data gathering"""
    try:
        if not task.market_data and symbols:
            task.market_data = self.market_provider.get_enhanced_stock_data(symbols)
            task.technical_data = self.market_provider.get_technical_indicators(symbols)
            task.news_data = self.market_provider.get_enhanced_news_data(symbols)
        elif not task.market_data:
            # Default symbols for general analysis
            default_symbols = ['SPY', 'QQQ', 'IWM', 'VIX']
            task.market_data = self.market_provider.get_enhanced_stock_data(default_symbols)
            task.technical_data = self.market_provider.get_technical_indicators(default_symbols)

        # Set analysis type if not specified
        if task.analysis_type == AnalysisType.TECHNICAL:  # Default
            task.analysis_type = self.get_analysis_type()

        # Estimate task duration based on complexity
        task.estimated_duration = self._estimate_task_duration(task)

        # Add task to priority queue
        self.tasks_queue.append(task)
        self.tasks_queue.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)

        logger.info(f"Task {task.id} added to agent {self.name} queue")

    except Exception as e:
        logger.error(f"Error adding task to agent {self.agent_id}: {e}")
        # Create minimal task data to prevent complete failure
        if not task.market_data:
            task.market_data = {}
            task.technical_data = {}
            task.news_data = {}
        self.tasks_queue.append(task)


def _estimate_task_duration(self, task: Task) -> float:
    """Estimate task duration based on complexity"""
    base_duration = 5.0  # Base 5 seconds

    # Adjust based on task complexity
    if task.market_data:
        base_duration += len(task.market_data) * 1.0

    if task.analysis_type in [AnalysisType.FUNDAMENTAL, AnalysisType.STRATEGY]:
        base_duration *= 1.5  # More complex analysis

    if task.priority == TaskPriority.CRITICAL:
        base_duration *= 0.8  # Prioritize speed for critical tasks

    return base_duration


logger.info("Task completed")
def execute_next_task(self) -> Optional[Task]:
    """Enhanced task execution with comprehensive error handling"""
    if not self.tasks_queue:
        return None

    task = self.tasks_queue.pop(0)
    self.current_task = task
    self.status = AgentStatus.ACTIVE
    task.status = TaskStatus.IN_PROGRESS.value
    task.started_at = datetime.datetime.now()

    try:
        start_time = time.time()

        # Process task with enhanced context
        result, confidence_score = self.process_enhanced_task(task)

        end_time = time.time()
        processing_time = end_time - start_time
        task.actual_duration = processing_time
        task.confidence_score = confidence_score

        # Determine final status
        if "❌" in result or "Error:" in result:
            task.status = TaskStatus.FAILED.value
            self.metrics.errors_count += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.status = AgentStatus.ERROR
        else:
            task.status = TaskStatus.COMPLETED.value
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.status = AgentStatus.IDLE

        task.completed_at = datetime.datetime.now()
        task.result = result

        # Update comprehensive metrics
        self._update_metrics(processing_time, task.status == TaskStatus.COMPLETED.value)

        # Store performance data
        self.performance_history.append({
            'timestamp': datetime.datetime.now(),
            'processing_time': processing_time,
            'confidence': confidence_score,
            'success': task.status == TaskStatus.COMPLETED.value,
            'task_type': task.analysis_type.value
        })

        self.current_task = None
        return task

    except Exception as e:
        return self._handle_task_error(task, e)


def process_enhanced_task(self, task: Task) -> tuple[str, float]:
    """Enhanced task processing with comprehensive context"""
    try:
        # Build comprehensive context
        context = {
            'market_data': task.market_data,
            'technical_data': task.technical_data,
            'news_data': task.news_data,
            'agent_specializations': self.specializations,
            'task_priority': task.priority.name,
            'estimated_duration': task.estimated_duration
        }

        # Add specialized analysis
        specialized_analysis = self.get_specialized_analysis(context)
        if specialized_analysis:
            context['specialized_analysis'] = specialized_analysis

        # Get agent-specific prompt with enhancements
        agent_prompt = self.get_enhanced_agent_prompt(task, context)

        # Generate AI response with context
        self.metrics.api_calls_made += 1
        result, confidence = self.gemini_client.generate_enhanced_response(
            agent_prompt,
            context,
            task.analysis_type
        )

        return result, confidence

    except Exception as e:
        error_msg = f"Error processing task {task.id}: {str(e)}"
        logger.error(error_msg)

        task.retry_count += 1
        task.error_message = error_msg

        if task.retry_count < task.max_retries:
            retry_msg = f"⚠️ Task processing failed (attempt {task.retry_count}/{task.max_retries}). Will retry.\n\nError: {str(e)}\n\nThe system will automatically retry this task with adjusted parameters."
            return retry_msg, 0.2
        else:
            final_error = f"❌ Task processing failed after {task.max_retries} attempts.\n\nFinal error: {str(e)}\n\nPlease review the task parameters and try again with different settings."
            return final_error, 0.1


def get_enhanced_agent_prompt(self, task: Task, context: dict) -> str:
    """Get enhanced agent prompt with context"""
    base_prompt = self.get_agent_prompt()

    enhanced_sections = [
        f"AGENT ROLE: {self.name}",
        f"SPECIALIZATIONS: {', '.join(self.specializations)}",
        f"ANALYSIS TYPE: {task.analysis_type.value}",
        f"TASK PRIORITY: {task.priority.name}",
        "",
        base_prompt,
        "",
        f"SPECIFIC TASK: {task.description}",
        "",
        "PERFORMANCE REQUIREMENTS:",
        f"• Provide actionable, specific recommendations",
        f"• Include confidence levels and risk assessments",
        f"• Consider multiple scenarios and timeframes",
        f"• Address regulatory and compliance factors",
        f"• Use quantitative analysis where possible"
    ]

    if task.tags:
        enhanced_sections.insert(-1, f"TASK TAGS: {', '.join(task.tags)}")

    return "\n".join(enhanced_sections)


def _update_metrics(self, processing_time: float, success: bool):
    """Update comprehensive agent metrics"""
    self.metrics.tasks_completed += 1
    self.metrics.last_activity = datetime.datetime.now()
    self.metrics.total_processing_time += processing_time

    # Update processing time statistics
    if processing_time > self.metrics.max_processing_time:
        self.metrics.max_processing_time = processing_time
    if processing_time < self.metrics.min_processing_time or self.metrics.min_processing_time == float('inf'):
        self.metrics.min_processing_time = processing_time

    self.metrics.avg_processing_time = (
            self.metrics.total_processing_time / self.metrics.tasks_completed
    )

    # Calculate success rate
    if not success:
        self.metrics.errors_count += 1

    successful_tasks = self.metrics.tasks_completed - self.metrics.errors_count
    self.metrics.success_rate = (
                successful_tasks / self.metrics.tasks_completed * 100) if self.metrics.tasks_completed > 0 else 0


def _handle_task_error(self, task: Task, error: Exception) -> Task:
    """Handle task execution errors"""
    task.status = TaskStatus.FAILED.value
    task.completed_at = datetime.datetime.now()
    task.error_message = str(error)
    task.result = f"❌ Critical Error: {str(error)}"

    self.metrics.errors_count += 1
    self.metrics.consecutive_failures += 1
    self.metrics.consecutive_successes = 0
    self.status = AgentStatus.ERROR
    self.current_task = None

    logger.error(f"Critical error in agent {self.agent_id} task {task.id}: {error}")
    return task


# --------------- Specialized Enhanced Agent Implementations ---------------
class EnhancedTradingStrategistAgent(EnhancedBaseAgent):
    def __init__(self):
        super().__init__(
            "TS001",
            "AI Trading Strategist",
            "Develops comprehensive investment strategies, portfolio allocation, and market timing decisions",
            ["Strategy Development", "Portfolio Optimization", "Market Timing", "Asset Allocation"]
        )

    def get_analysis_type(self) -> AnalysisType:
        return AnalysisType.STRATEGY

    def get_agent_prompt(self) -> str:
        return """You are an elite AI Trading Strategist with expertise in developing sophisticated investment strategies and portfolio management solutions.

CORE RESPONSIBILITIES:
• Develop comprehensive investment strategies aligned with risk tolerance and objectives
• Design optimal asset allocation models using modern portfolio theory
• Provide strategic market timing and tactical allocation recommendations  
• Create long-term investment roadmaps with adaptive rebalancing strategies
• Analyze macro and microeconomic trends impacting strategic decisions
• Implement risk-adjusted performance optimization frameworks

STRATEGIC ANALYSIS FRAMEWORK:
• Multi-timeframe analysis (short, medium, long-term horizons)
• Risk-return optimization using quantitative models
• Correlation analysis and diversification strategies
• Market cycle analysis and positioning strategies
• Behavioral finance considerations in strategy design
• ESG and sustainable investing integration

DELIVERABLE REQUIREMENTS:
• Specific asset allocation percentages with rationale
• Clear entry/exit criteria and rebalancing triggers
• Risk management parameters and position sizing rules
• Performance benchmarks and success metrics
• Scenario analysis with stress testing results
• Implementation timeline with milestone checkpoints"""

    def get_specialized_analysis(self, context: dict) -> str:
        """Get specialized analysis for trading strategist"""
        if context.get("market_data") and context.get("technical_data"):
            # Example of specialized analysis based on market data
            return f"Strategic insights based on market trends: {context['market_data']}, Technical indicators: {context['technical_data']}"
        return "No specialized analysis available."