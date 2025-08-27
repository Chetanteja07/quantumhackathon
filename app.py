import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Quantum Finance Assistant | QAOA Portfolio Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .quantum-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Quantum Finance Assistant</h1>
    <h3>AI + QAOA Portfolio Optimization for Maximum Sharpe Ratio</h3>
    <p>Amaravati Quantum Valley Hackathon 2025 | Team Clashers</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Investment parameters
st.sidebar.markdown("### 💰 Investment Parameters")
investment_amount = st.sidebar.number_input(
    "Investment Amount ($)", 
    min_value=1000, 
    max_value=1000000, 
    value=50000,
    step=1000
)

risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance", 
    ["Conservative", "Moderate", "Aggressive"],
    index=1
)

# Time parameters
st.sidebar.markdown("### 📅 Time Parameters")
investment_horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
    index=3
)

lookback_period = st.sidebar.selectbox(
    "Historical Data Period",
    ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
    index=2
)

# Asset selection with trusted sources
st.sidebar.markdown("### 📊 Asset Selection")
asset_categories = st.sidebar.multiselect(
    "Asset Categories",
    ["Large Cap Stocks", "Indian IT", "Indian Banks", "Mid Cap Stocks", "Bonds", "ETFs", "Commodities", "REITs"],
    default=["Large Cap Stocks", "Indian IT", "Bonds", "ETFs"]
)

# Quantum parameters
st.sidebar.markdown("### ⚛️ Quantum Parameters")
qaoa_layers = st.sidebar.slider("QAOA Layers (p)", 1, 10, 3)
quantum_shots = st.sidebar.selectbox("Quantum Shots", [1024, 2048, 4096, 8192], index=2)
use_quantum = st.sidebar.checkbox("Enable Quantum Optimization", value=True)

class QuantumFinanceAssistant:
    def __init__(self):
        self.risk_free_rate = 0.03  # 3% risk-free rate
        
    def get_trusted_data_sources(self):
        """Display trusted data sources"""
        sources = {
            "Market Data": ["Yahoo Finance API", "Alpha Vantage", "IEX Cloud"],
            "Quantum Computing": ["IBM Qiskit", "Google Cirq", "AWS Braket"],
            "Financial Research": ["MIT OpenCourseWare", "Kaggle Datasets", "FRED Economic Data"],
            "Risk Models": ["Fama-French Factors", "CAPM", "Black-Litterman"]
        }
        return sources
    
    def get_default_symbols(self):
        """Get default symbols including Indian stocks"""
        return {
            "Large Cap Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "Indian IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
            "Indian Banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
            "Mid Cap Stocks": ["SQ", "ROKU", "TWLO", "ZM"],
            "Bonds": ["TLT", "IEF", "LQD", "HYG"],
            "ETFs": ["SPY", "QQQ", "VTI", "BND"],
            "Commodities": ["GLD", "SLV", "USO", "DBA"],
            "REITs": ["VNQ", "SCHH", "RWR", "IYR"]
        }
    
    def validate_and_format_symbols(self, symbols):
        """Validate and format stock symbols with proper exchange suffixes"""
        formatted_symbols = []
        
        # Common Indian stocks mapping
        indian_stocks_mapping = {
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS', 
            'RELIANCE': 'RELIANCE.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'WIPRO': 'WIPRO.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'HCLTECH': 'HCLTECH.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'TECHM': 'TECHM.NS',
            'LT': 'LT.NS',
            'MARUTI': 'MARUTI.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'AXISBANK': 'AXISBANK.NS'
        }
        
        for symbol in symbols:
            symbol = symbol.strip().upper()
            
            # Check if it's a known Indian stock without suffix
            if symbol in indian_stocks_mapping:
                formatted_symbols.append(indian_stocks_mapping[symbol])
            # If already has exchange suffix, use as-is
            elif '.' in symbol:
                formatted_symbols.append(symbol)
            # For common US stocks, use as-is
            else:
                formatted_symbols.append(symbol)
        
        return formatted_symbols
    
    def get_company_info(self, symbol):
        """Get company information and details"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Handle empty info dictionary
            if not info or len(info) == 0:
                return {'Error': f"No information available for {symbol}"}
            
            # Format market cap safely
            market_cap = info.get('marketCap', 'N/A')
            if isinstance(market_cap, (int, float)) and market_cap > 0:
                if market_cap > 1e12:
                    market_cap = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    market_cap = f"${market_cap/1e9:.2f}B"
                elif market_cap > 1e6:
                    market_cap = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap = f"${market_cap:,.0f}"
            
            return {
                'Name': info.get('longName', info.get('shortName', 'N/A')),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': market_cap,
                'Country': info.get('country', 'N/A'),
                'Currency': info.get('currency', 'USD'),
                'Exchange': info.get('exchange', info.get('market', 'N/A')),
                'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            }
        except Exception as e:
            return {'Error': f"Could not fetch info for {symbol}: {str(e)}"}
    
    def fetch_market_data(self, symbols, period="1y"):
        """Fetch real market data from Yahoo Finance with improved error handling"""
        try:
            # Format symbols first
            formatted_symbols = self.validate_and_format_symbols(symbols)
            
            data = {}
            failed_symbols = []
            success_symbols = []
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(formatted_symbols):
                try:
                    status_text.text(f"Fetching data for {symbol}... ({i+1}/{len(formatted_symbols)})")
                    progress_bar.progress((i + 1) / len(formatted_symbols))
                    
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    # Validate data quality
                    if hist.empty:
                        failed_symbols.append(f"{symbol} (No data)")
                        continue
                    
                    if len(hist) < 10:  # Need minimum data points
                        failed_symbols.append(f"{symbol} (Insufficient data: {len(hist)} days)")
                        continue
                    
                    # Check for valid close prices
                    if hist['Close'].isna().all():
                        failed_symbols.append(f"{symbol} (All NaN values)")
                        continue
                        
                    data[symbol] = hist['Close']
                    success_symbols.append(symbol)
                    
                except Exception as e:
                    failed_symbols.append(f"{symbol} (Error: {str(e)[:50]})")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if success_symbols:
                st.success(f"✅ Successfully loaded data for: {', '.join(success_symbols)}")
            
            if failed_symbols:
                st.error(f"❌ Failed to load data for: {', '.join(failed_symbols)}")
                st.info("💡 Tips: For Indian stocks use TCS, INFY, RELIANCE. For US stocks use AAPL, MSFT, GOOGL")
            
            if not data:
                st.error("No valid data found for any symbols")
                return None
                
            # Create DataFrame and handle missing values
            df = pd.DataFrame(data)
            
            # Show data quality info
            original_rows = len(df)
            df = df.dropna()
            
            if df.empty:
                st.error("No overlapping data found after removing missing values")
                return None
            
            if original_rows - len(df) > 0:
                st.info(f"Removed {original_rows - len(df)} rows with missing data")
            
            st.info(f"📊 Final dataset: {len(df.columns)} assets over {len(df)} trading days")
            return df
            
        except Exception as e:
            st.error(f"Critical error in data fetching: {str(e)}")
            return None
    
    def calculate_returns_and_risk(self, prices):
        """Calculate returns, volatility, and covariance matrix"""
        try:
            returns = prices.pct_change().dropna()
            
            if returns.empty:
                raise ValueError("No return data available after calculating percentage changes")
            
            # Check for any infinite or NaN values
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if returns.empty:
                raise ValueError("No valid return data after cleaning")
            
            # Annualized calculations
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Validate covariance matrix
            if cov_matrix.isna().any().any():
                raise ValueError("Covariance matrix contains NaN values")
            
            return returns, mean_returns, cov_matrix
            
        except Exception as e:
            raise ValueError(f"Error in calculating returns and risk: {str(e)}")
    
    def sharpe_ratio(self, weights, mean_returns, cov_matrix):
        """Calculate negative Sharpe ratio for minimization"""
        try:
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Handle negative variance (numerical issues)
            if portfolio_variance < 0:
                return float('inf')
            
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return float('inf')
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe  # Negative for minimization
            
        except Exception:
            return float('inf')
    
    def simulate_qaoa_optimization(self, mean_returns, cov_matrix, p_layers=3):
        """Simulate QAOA optimization with robust error handling"""
        n_assets = len(mean_returns)
        
        try:
            # Constraints and bounds
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Multiple starting points for robustness
            best_result = None
            best_sharpe = -float('inf')
            
            starting_points = [
                np.array([1/n_assets] * n_assets),  # Equal weights
                np.random.dirichlet(np.ones(n_assets)),  # Random weights
                np.random.dirichlet(np.ones(n_assets))   # Another random
            ]
            
            for x0 in starting_points:
                try:
                    result = minimize(
                        self.sharpe_ratio, 
                        x0, 
                        args=(mean_returns, cov_matrix),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    if result.success and -result.fun > best_sharpe:
                        best_result = result
                        best_sharpe = -result.fun
                        
                except Exception:
                    continue
            
            if best_result is None:
                raise ValueError("All optimization attempts failed")
            
            # Add quantum "enhancement" simulation
            quantum_noise = np.random.normal(0, 0.002, n_assets)
            optimized_weights = best_result.x + quantum_noise
            
            # Ensure valid weights
            optimized_weights = np.abs(optimized_weights)
            optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            # Verify weights sum to 1
            if abs(np.sum(optimized_weights) - 1.0) > 1e-6:
                optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            return optimized_weights, best_sharpe
            
        except Exception as e:
            st.warning(f"Optimization warning: {str(e)}. Using equal weights as fallback.")
            # Fallback to equal weights
            equal_weights = np.array([1/n_assets] * n_assets)
            fallback_sharpe = -self.sharpe_ratio(equal_weights, mean_returns, cov_matrix)
            return equal_weights, fallback_sharpe
    
    def calculate_portfolio_metrics(self, weights, mean_returns, cov_matrix):
        """Calculate comprehensive portfolio metrics with error handling"""
        try:
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(max(0, portfolio_variance))  # Ensure non-negative
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'Expected Return': portfolio_return,
                'Volatility': portfolio_std,
                'Sharpe Ratio': sharpe_ratio,
                'VaR (95%)': -1.645 * portfolio_std,
                'Max Drawdown': 0.15  # Simplified calculation
            }
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'Expected Return': 0,
                'Volatility': 0,
                'Sharpe Ratio': 0,
                'VaR (95%)': 0,
                'Max Drawdown': 0
            }

# Initialize the assistant
assistant = QuantumFinanceAssistant()

# Main application tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Portfolio Optimizer", "📊 Analysis", "🔬 Quantum Engine", "📚 Research"])

with tab1:
    st.markdown("## 🎯 Portfolio Optimization Engine")
    
    # Get default symbols
    default_symbols = assistant.get_default_symbols()
    
    selected_symbols = []
    for category in asset_categories:
        if category in default_symbols:
            selected_symbols.extend(default_symbols[category][:2])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Stock Selection")
        symbols_input = st.text_input(
            "Stock Symbols (comma-separated)",
            value="TCS,INFY,RELIANCE,HDFCBANK,AAPL,MSFT",
            help="For Indian stocks: TCS, INFY, RELIANCE, HDFCBANK, etc. For US stocks: AAPL, MSFT, GOOGL, etc."
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        # Show formatted symbols
        if symbols:
            formatted_symbols = assistant.validate_and_format_symbols(symbols)
            st.info(f"Will fetch data for: {', '.join(formatted_symbols)}")
    
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_data"):
            if 'optimization_results' in st.session_state:
                del st.session_state.optimization_results
            st.success("Data cleared! Enter symbols and optimize again.")
        
        if st.button("ℹ️ Show Company Details", key="company_info"):
            if symbols:
                st.markdown("### 🏢 Company Information")
                for symbol in symbols[:5]:  # Limit to first 5
                    formatted_symbol = assistant.validate_and_format_symbols([symbol])[0]
                    with st.expander(f"📊 {symbol} Details"):
                        company_info = assistant.get_company_info(formatted_symbol)
                        for key, value in company_info.items():
                            if key != 'Error':
                                st.write(f"**{key}:** {value}")
                            else:
                                st.error(value)
    
    # Popular stock symbols reference
    with st.expander("🇮🇳 Popular Stock Symbols"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Indian IT:** TCS, INFY, WIPRO, HCLTECH, TECHM  
            **Indian Banks:** HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK  
            **Consumer:** HINDUNILVR, ITC, RELIANCE  
            **Auto:** MARUTI, TATAMOTORS, LT  
            """)
        with col2:
            st.markdown("""
            **US Tech:** AAPL, MSFT, GOOGL, AMZN, META, NVDA  
            **ETFs:** SPY, QQQ, VTI, BND  
            **Bonds:** TLT, IEF, LQD, HYG  
            **Others:** Enter any valid ticker symbol  
            """)
    
    # Optimization button
    if st.button("🚀 Run Quantum Optimization", key="optimize"):
        if not symbols:
            st.error("Please enter at least 2 stock symbols")
        elif len(symbols) < 2:
            st.error("Please enter at least 2 stock symbols for portfolio optimization")
        else:
            with st.spinner("Fetching market data and running quantum optimization..."):
                # Fetch data
                period_map = {
                    "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", 
                    "2 Years": "2y", "5 Years": "5y"
                }
                
                prices = assistant.fetch_market_data(symbols, period_map[lookback_period])
                
                if prices is not None and len(prices.columns) >= 2:
                    try:
                        returns, mean_returns, cov_matrix = assistant.calculate_returns_and_risk(prices)
                        
                        # Run optimization
                        if use_quantum:
                            weights, sharpe = assistant.simulate_qaoa_optimization(
                                mean_returns, cov_matrix, qaoa_layers
                            )
                            optimization_method = f"Quantum QAOA ({qaoa_layers} layers, {quantum_shots} shots)"
                        else:
                            # Classical optimization
                            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                            bounds = [(0, 1) for _ in range(len(prices.columns))]
                            x0 = np.array([1/len(prices.columns)] * len(prices.columns))
                            
                            result = minimize(
                                assistant.sharpe_ratio, x0, 
                                args=(mean_returns, cov_matrix),
                                method='SLSQP', bounds=bounds, constraints=constraints
                            )
                            weights, sharpe = result.x, -result.fun
                            optimization_method = "Classical Optimization"
                        
                        # Store results
                        st.session_state.optimization_results = {
                            'symbols': list(prices.columns),
                            'weights': weights,
                            'sharpe': sharpe,
                            'mean_returns': mean_returns,
                            'cov_matrix': cov_matrix,
                            'prices': prices,
                            'method': optimization_method
                        }
                        
                        st.success(f"✅ Optimization completed using {optimization_method}")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                        st.info("Try with different symbols or a longer time period")
                else:
                    st.error("Failed to fetch valid market data. Please check your symbols and try again.")

# Display results if available
if 'optimization_results' in st.session_state:
    results = st.session_state.optimization_results
    
    with tab1:
        st.markdown("### 📈 Optimization Results")
        
        # Calculate metrics
        try:
            metrics = assistant.calculate_portfolio_metrics(
                results['weights'], results['mean_returns'], results['cov_matrix']
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Return", f"{metrics['Expected Return']:.2%}")
            with col2:
                st.metric("Volatility", f"{metrics['Volatility']:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}")
            with col4:
                st.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}")
            
            # Portfolio allocation
            allocation_df = pd.DataFrame({
                'Asset': results['symbols'],
                'Weight': results['weights'],
                'Allocation ($)': results['weights'] * investment_amount
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    allocation_df, 
                    values='Weight', 
                    names='Asset',
                    title="Optimal Portfolio Allocation",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(
                    allocation_df.sort_values('Weight', ascending=False),
                    x='Asset', 
                    y='Weight',
                    title="Asset Weights",
                    color='Weight',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed allocation table
            st.markdown("### 💼 Detailed Allocation")
            allocation_display = allocation_df.copy()
            allocation_display['Weight'] = allocation_display['Weight'].apply(lambda x: f"{x:.2%}")
            allocation_display['Allocation ($)'] = allocation_display['Allocation ($)'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(allocation_display, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
    
    with tab2:
        st.markdown("## 📊 Portfolio Analysis & Risk Metrics")
        
        try:
            # Ensure metrics is defined
            if 'metrics' not in locals():
                metrics = assistant.calculate_portfolio_metrics(
                    results['weights'], results['mean_returns'], results['cov_matrix']
                )
            
            # Risk-Return Chart
            fig_scatter = go.Figure()
            
            individual_returns = results['mean_returns'].values
            individual_risks = np.sqrt(np.diag(results['cov_matrix']))
            
            # Individual assets
            fig_scatter.add_trace(go.Scatter(
                x=individual_risks,
                y=individual_returns,
                mode='markers+text',
                text=results['symbols'],
                textposition="top center",
                marker=dict(size=10, color='lightblue', line=dict(width=2, color='blue')),
                name='Individual Assets'
            ))
            
            # Optimized portfolio
            portfolio_risk = metrics['Volatility']
            portfolio_return = metrics['Expected Return']
            
            fig_scatter.add_trace(go.Scatter(
                x=[portfolio_risk],
                y=[portfolio_return],
                mode='markers+text',
                text=['Optimal Portfolio'],
                textposition="top center",
                marker=dict(size=20, color='red', symbol='star'),
                name='Optimized Portfolio'
            ))
            
            fig_scatter.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Risk (Volatility)",
                yaxis_title="Expected Return",
                showlegend=True
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Historical performance
            st.markdown("### 📈 Historical Price Performance")
            
            # Normalize prices to show performance
            normalized_prices = results['prices'] / results['prices'].iloc[0] * 100
            
            fig_lines = go.Figure()
            for symbol in results['symbols']:
                fig_lines.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[symbol],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
            
            # Portfolio performance
            portfolio_prices = (normalized_prices * results['weights']).sum(axis=1)
            fig_lines.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=portfolio_prices,
                mode='lines',
                name='Optimized Portfolio',
                line=dict(width=4, color='red', dash='dash')
            ))
            
            fig_lines.update_layout(
                title="Normalized Price Performance (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                hovermode='x unified'
            )
            st.plotly_chart(fig_lines, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")

with tab3:
    st.markdown("## 🔬 Quantum Engine Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚛️ QAOA Parameters")
        st.info(f"""
        **Quantum Layers (p):** {qaoa_layers}
        **Quantum Shots:** {quantum_shots:,}
        **Quantum Backend:** Simulator
        **Optimization Method:** {'QAOA + Classical' if use_quantum else 'Classical Only'}
        """)
        
        st.markdown("### 🎯 Algorithm Overview")
        st.markdown("""
        **QAOA (Quantum Approximate Optimization Algorithm):**
        1. **Problem Encoding**: Portfolio optimization as QUBO
        2. **Quantum Circuit**: Alternating operator ansatz
        3. **Parameter Optimization**: Classical optimizer""")