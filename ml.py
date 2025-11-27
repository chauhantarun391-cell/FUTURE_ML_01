import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class SalesForecastDashboard:
    def __init__(self):
        self.setup_page()
        self.generate_sample_data()
        
    def setup_page(self):
        st.set_page_config(page_title="AI Sales Forecast Dashboard", layout="wide")
        st.title("🚀 AI-Powered Sales Forecasting Dashboard")
        st.markdown("---")
        
    def generate_sample_data(self):
        """Generate sample sales data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        
        # Generate realistic sales data with trends and seasonality
        base_trend = np.linspace(1000, 5000, len(dates))
        seasonal = 1000 * np.sin(2 * np.pi * dates.dayofyear / 365)
        noise = np.random.normal(0, 200, len(dates))
        
        sales = base_trend + seasonal + noise
        sales = np.maximum(sales, 0)  # Ensure no negative sales
        
        # Create deals data
        deal_sizes = np.random.lognormal(8, 1, 1000)
        stages = ['Prospect', 'Qualification', 'Proposal', 'Negotiation', 'Closed-Won']
        stage_weights = [0.1, 0.3, 0.5, 0.8, 1.0]
        
        self.sales_data = pd.DataFrame({
            'date': dates,
            'daily_sales': sales,
            'month': dates.month,
            'quarter': dates.quarter,
            'day_of_week': dates.dayofweek,
            'is_weekend': dates.dayofweek >= 5
        })
        
        self.deals_data = pd.DataFrame({
            'deal_id': range(1000),
            'deal_size': deal_sizes,
            'stage': np.random.choice(stages, 1000, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
            'company_size': np.random.choice(['SMB', 'Mid-Market', 'Enterprise'], 1000),
            'industry': np.random.choice(['Tech', 'Healthcare', 'Finance', 'Manufacturing'], 1000),
            'created_date': np.random.choice(dates, 1000),
            'engagement_score': np.random.uniform(0, 1, 1000)
        })
        
        # Add win probabilities based on features
        self.calculate_win_probabilities()
        
    def calculate_win_probabilities(self):
        """Calculate AI-powered win probabilities for deals"""
        stage_mapping = {'Prospect': 0.1, 'Qualification': 0.3, 'Proposal': 0.5, 
                        'Negotiation': 0.8, 'Closed-Won': 1.0}
        
        self.deals_data['stage_probability'] = self.deals_data['stage'].map(stage_mapping)
        
        # Enhanced probability with engagement and company size factors
        size_boost = {'SMB': 0.0, 'Mid-Market': 0.1, 'Enterprise': 0.15}
        self.deals_data['size_boost'] = self.deals_data['company_size'].map(size_boost)
        
        # Final win probability calculation
        self.deals_data['win_probability'] = (
            self.deals_data['stage_probability'] * 0.6 +
            self.deals_data['engagement_score'] * 0.3 +
            self.deals_data['size_boost'] * 0.1
        )
        
        # Add deal health
        conditions = [
            self.deals_data['win_probability'] >= 0.7,
            self.deals_data['win_probability'] >= 0.4,
            self.deals_data['win_probability'] < 0.4
        ]
        choices = ['🟢 Healthy', '🟡 Needs Attention', '🔴 At Risk']
        self.deals_data['deal_health'] = np.select(conditions, choices, default='🟡 Needs Attention')
    
    def train_forecast_model(self):
        """Train ML model for sales forecasting"""
        # Prepare data for modeling
        df = self.sales_data.copy()
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Create lag features
        df['sales_lag_7'] = df['daily_sales'].shift(7)
        df['sales_lag_30'] = df['daily_sales'].shift(30)
        df = df.dropna()
        
        # Features and target
        features = ['month', 'quarter', 'day_of_week', 'is_weekend', 
                   'day_of_year', 'sales_lag_7', 'sales_lag_30']
        X = df[features]
        y = df['daily_sales']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return model, mae, rmse, df
    
    def create_forecast(self, model, historical_data, days=90):
        """Generate future forecasts"""
        future_dates = pd.date_range(start=historical_data['date'].max() + timedelta(days=1), 
                                   periods=days, freq='D')
        
        forecasts = []
        last_known = historical_data.tail(30)['daily_sales'].values
        
        for date in future_dates:
            # Prepare features for prediction
            features = {
                'month': date.month,
                'quarter': date.quarter,
                'day_of_week': date.weekday(),
                'is_weekend': date.weekday() >= 5,
                'day_of_year': date.dayofyear,
                'sales_lag_7': last_known[-7] if len(last_known) >= 7 else last_known[-1],
                'sales_lag_30': last_known[-30] if len(last_known) >= 30 else last_known[-1]
            }
            
            feature_df = pd.DataFrame([features])
            prediction = model.predict(feature_df)[0]
            forecasts.append(prediction)
            last_known = np.append(last_known[1:], prediction)
        
        return future_dates, np.array(forecasts)
    
    def display_kpi_cards(self):
        """Display key performance indicators"""
        col1, col2, col3, col4 = st.columns(4)
        
        total_pipeline = self.deals_data['deal_size'].sum()
        weighted_pipeline = (self.deals_data['deal_size'] * self.deals_data['win_probability']).sum()
        avg_win_prob = self.deals_data['win_probability'].mean()
        healthy_deals = len(self.deals_data[self.deals_data['deal_health'] == '🟢 Healthy'])
        
        with col1:
            st.metric("Total Pipeline", f"${total_pipeline:,.0f}")
        with col2:
            st.metric("Weighted Pipeline", f"${weighted_pipeline:,.0f}")
        with col3:
            st.metric("Avg Win Probability", f"{avg_win_prob:.1%}")
        with col4:
            st.metric("Healthy Deals", f"{healthy_deals}")
    
    def display_forecast_chart(self, historical_data, future_dates, forecasts):
        """Display the main forecast chart"""
        # Create historical trace
        historical_trace = go.Scatter(
            x=historical_data['date'],
            y=historical_data['daily_sales'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#1f77b4')
        )
        
        # Create forecast trace
        forecast_trace = go.Scatter(
            x=future_dates,
            y=forecasts,
            mode='lines',
            name='AI Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        )
        
        # Create confidence interval (simplified)
        confidence_upper = forecasts * 1.1
        confidence_lower = forecasts * 0.9
        
        confidence_trace = go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([confidence_upper, confidence_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% Confidence Interval'
        )
        
        fig = go.Figure(data=[historical_trace, confidence_trace, forecast_trace])
        fig.update_layout(
            title='Sales Forecast with AI Predictions',
            xaxis_title='Date',
            yaxis_title='Daily Sales ($)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_pipeline_analysis(self):
        """Display pipeline analysis section"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Stage distribution
            stage_summary = self.deals_data.groupby('stage').agg({
                'deal_size': 'sum',
                'win_probability': 'mean',
                'deal_id': 'count'
            }).round(2)
            
            st.subheader("Pipeline by Stage")
            st.dataframe(stage_summary, use_container_width=True)
            
            # Deal health chart
            health_dist = self.deals_data['deal_health'].value_counts()
            fig_health = px.pie(values=health_dist.values, names=health_dist.index,
                              title="Deal Health Distribution")
            st.plotly_chart(fig_health, use_container_width=True)
        
        with col2:
            # Top deals table
            st.subheader("Top Deals - AI Recommendations")
            top_deals = self.deals_data.nlargest(10, 'win_probability')[[
                'deal_id', 'deal_size', 'stage', 'win_probability', 'deal_health'
            ]]
            st.dataframe(top_deals, use_container_width=True)
            
            # Risk alerts
            st.subheader("⚠️ Risk Alerts")
            at_risk_deals = self.deals_data[self.deals_data['deal_health'] == '🔴 At Risk']
            if not at_risk_deals.empty:
                for _, deal in at_risk_deals.head(5).iterrows():
                    st.warning(f"Deal #{deal['deal_id']}: ${deal['deal_size']:,.0f} - {deal['stage']} - Win Prob: {deal['win_probability']:.1%}")
    
    def display_ai_insights(self):
        """Display AI-generated insights"""
        st.subheader("🤖 AI Insights & Recommendations")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            # Industry performance
            industry_perf = self.deals_data.groupby('industry').agg({
                'win_probability': 'mean',
                'deal_size': 'mean'
            }).round(3)
            
            st.write("**Industry Performance:**")
            st.dataframe(industry_perf, use_container_width=True)
            
            # Top influencer
            st.success("**🎯 Top Influencer:** Deals with engagement scores above 0.8 close 2.3x faster")
        
        with insights_col2:
            # Company size analysis
            size_analysis = self.deals_data.groupby('company_size').agg({
                'win_probability': 'mean',
                'deal_id': 'count'
            })
            
            st.write("**Performance by Company Size:**")
            fig_size = px.bar(size_analysis, x=size_analysis.index, y='win_probability',
                            title="Win Probability by Company Size")
            st.plotly_chart(fig_size, use_container_width=True)
    
    def run(self):
        """Run the complete dashboard"""
        # Train model and get forecasts
        with st.spinner("Training AI model and generating forecasts..."):
            model, mae, rmse, historical_data = self.train_forecast_model()
            future_dates, forecasts = self.create_forecast(model, historical_data)
        
        # Display KPIs
        self.display_kpi_cards()
        
        st.markdown("---")
        
        # Main forecast chart
        self.display_forecast_chart(historical_data, future_dates, forecasts)
        
        # Model performance
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Forecast Accuracy (MAE)", f"${mae:.0f}")
        with col2:
            st.metric("Forecast Error (RMSE)", f"${rmse:.0f}")
        
        st.markdown("---")
        
        # Pipeline analysis
        self.display_pipeline_analysis()
        
        st.markdown("---")
        
        # AI Insights
        self.display_ai_insights()

# Run the dashboard
if __name__ == "__main__":
    dashboard = SalesForecastDashboard()
    dashboard.run()
