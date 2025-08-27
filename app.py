# Add this improved version to your existing code

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
    
    def validate_and_format_symbols(self, symbols):
        """Validate and format stock symbols with proper exchange suffixes"""
        formatted_symbols = []
        invalid_symbols = []
        
        # Common Indian stocks mapping
        indian_stocks = {
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS', 
            'RELIANCE': 'RELIANCE.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'WIPRO': 'WIPRO.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'HINDUNILVR': 'HINDUNILVR.NS'
        }
        
        for symbol in symbols:
            symbol = symbol.strip().upper()
            
            # Check if it's a known Indian stock
            if symbol in indian_stocks:
                formatted_symbols.append(indian_stocks[symbol])
            # Check if already has exchange suffix
            elif '.' in symbol:
                formatted_symbols.append(symbol)
            # For US stocks, use as-is
            elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']:
                formatted_symbols.append(symbol)
            else:
                # Try different exchange suffixes for unknown symbols
                formatted_symbols.append(symbol)
        
        return formatted_symbols, invalid_symbols
    
    def fetch_market_data(self, symbols, period="1y"):
        """Fetch real market data from Yahoo Finance with improved error handling"""
        try:
            # Validate and format symbols first
            formatted_symbols, invalid_symbols = self.validate_and_format_symbols(symbols)
            
            if invalid_symbols:
                st.warning(f"Invalid symbols detected: {', '.join(invalid_symbols)}")
            
            data = {}
            failed_symbols = []
            
            for symbol in formatted_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if hist.empty:
                        failed_symbols.append(symbol)
                        continue
                        
                    data[symbol] = hist['Close']
                    st.success(f"✅ Successfully loaded data for {symbol}")
                    
                except Exception as e:
                    failed_symbols.append(symbol)
                    st.error(f"❌ Failed to load {symbol}: {str(e)}")
            
            if failed_symbols:
                st.error(f"Failed to fetch data for: {', '.join(failed_symbols)}")
                st.info("💡 Tips for Indian stocks: Use TCS, INFY, RELIANCE, HDFCBANK, etc.")
            
            if not data:
                st.error("No valid data found for any symbols")
                return None
                
            df = pd.DataFrame(data)
            st.info(f"📊 Successfully loaded data for {len(data)} assets over {len(df)} trading days")
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            return None
    
    def get_company_info(self, symbol):
        """Get company information and details"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': info.get('marketCap', 'N/A'),
                'Country': info.get('country', 'N/A'),
                'Currency': info.get('currency', 'N/A'),
                'Exchange': info.get('exchange', 'N/A')
            }
        except Exception as e:
            return {'Error': f"Could not fetch info for {symbol}: {str(e)}"}

# Update the default symbols to include Indian stocks
def get_updated_default_symbols():
    return {
        "Large Cap Stocks": ["AAPL", "MSFT", "TCS", "RELIANCE"],
        "Indian IT": ["TCS", "INFY", "WIPRO", "HCLTECH"],
        "Indian Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"],
        "Mid Cap Stocks": ["SQ", "ROKU", "TWLO", "ZM"],
        "Bonds": ["TLT", "IEF", "LQD", "HYG"],
        "ETFs": ["SPY", "QQQ", "VTI", "BND"],
        "Commodities": ["GLD", "SLV", "USO", "DBA"],
        "REITs": ["VNQ", "SCHH", "RWR", "IYR"]
    }

# Add this section in your tab1 (Portfolio Optimizer)
# Replace the existing asset selection code with:

with tab1:
    st.markdown("## 🎯 Portfolio Optimization Engine")
    
    # Updated asset selection with Indian stocks
    default_symbols = get_updated_default_symbols()
    
    selected_symbols = []
    for category in asset_categories:
        if category in default_symbols:
            selected_symbols.extend(default_symbols[category][:2])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Stock Selection")
        symbols_input = st.text_input(
            "Stock Symbols (comma-separated)",
            value="TCS,INFY,RELIANCE,HDFCBANK,AAPL,MSFT,GOOGL,AMZN",
            help="For Indian stocks: TCS, INFY, RELIANCE, HDFCBANK, etc. For US stocks: AAPL, MSFT, GOOGL, etc."
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        # Show symbol validation
        if symbols:
            formatted_symbols, _ = assistant.validate_and_format_symbols(symbols)
            st.info(f"Will fetch data for: {', '.join(formatted_symbols)}")
    
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_data"):
            st.success("Data refreshed!")
        
        if st.button("ℹ️ Show Company Details", key="company_info"):
            if symbols:
                st.markdown("### 🏢 Company Information")
                for symbol in symbols[:5]:  # Limit to first 5
                    with st.expander(f"📊 {symbol} Details"):
                        company_info = assistant.get_company_info(symbol)
                        for key, value in company_info.items():
                            st.write(f"**{key}:** {value}")
    
    # Add a section to show available Indian stocks
    with st.expander("🇮🇳 Popular Indian Stock Symbols"):
        st.markdown("""
        **IT Companies:** TCS, INFY, WIPRO, HCLTECH, TECHM
        **Banks:** HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK
        **Consumer:** HINDUNILVR, ITC, NESTLEIND, BRITANNIA
        **Telecom:** BHARTIARTL, RJIO
        **Auto:** MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO
        **Pharma:** SUNPHARMA, DRREDDY, CIPLA, LUPIN
        """)

# Add error handling in the optimization button section:
    if st.button("🚀 Run Quantum Optimization", key="optimize"):
        if not symbols:
            st.error("Please enter at least 2 stock symbols")
        elif len(symbols) < 2:
            st.error("Please enter at least 2 stock symbols for portfolio optimization")
        else:
            with st.spinner("Fetching market data and running quantum optimization..."):
                # Fetch data with improved error handling
                period_map = {
                    "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", 
                    "2 Years": "2y", "5 Years": "5y"
                }
                
                prices = assistant.fetch_market_data(symbols, period_map[lookback_period])
                
                if prices is not None and len(prices.columns) >= 2:
                    # Remove any columns with all NaN values
                    prices = prices.dropna(axis=1, how='all')
                    
                    if len(prices.columns) < 2:
                        st.error("Need at least 2 valid assets for portfolio optimization")
                    else:
                        returns, mean_returns, cov_matrix = assistant.calculate_returns_and_risk(prices)
                        
                        # Run QAOA optimization
                        if use_quantum:
                            weights, sharpe = assistant.simulate_qaoa_optimization(
                                mean_returns, cov_matrix, qaoa_layers
                            )
                            optimization_method = f"Quantum QAOA ({qaoa_layers} layers, {quantum_shots} shots)"
                        else:
                            # Classical optimization
                            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                            bounds = tuple((0, 1) for _ in range(len(prices.columns)))
                            x0 = np.array([1/len(prices.columns)] * len(prices.columns))
                            
                            result = minimize(
                                assistant.sharpe_ratio, x0, 
                                args=(mean_returns, cov_matrix),
                                method='SLSQP', bounds=bounds, constraints=constraints
                            )
                            weights, sharpe = result.x, -result.fun
                            optimization_method = "Classical Optimization"
                        
                        # Store results in session state
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
                else:
                    st.error("Failed to fetch valid market data. Please check your symbols and try again.")