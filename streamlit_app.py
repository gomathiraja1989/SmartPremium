import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SmartPremium Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #fdfdff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    try:
        artifacts = joblib.load('outputs/model_artifacts.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please run the training pipeline first.")
        st.info("💡 Run 'python main.py' to train the model and generate artifacts.")
        return None

def predict_premium(input_data, artifacts):
    try:
        # Prepare input features
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col, encoder in artifacts['label_encoders'].items():
            if col in input_df.columns:
                # Handle unseen labels
                if input_data[col] in encoder.classes_:
                    input_df[col] = encoder.transform([input_data[col]])[0]
                else:
                    # Use the first class as default for unseen labels
                    input_df[col] = 0
        
        # Ensure all features are present
        for feature in artifacts['feature_columns']:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training
        input_df = input_df[artifacts['feature_columns']]
        
        # Scale features
        input_scaled = artifacts['scaler'].transform(input_df)
        
        # Make prediction
        prediction = artifacts['model'].predict(input_scaled)[0]
        
        return max(500, prediction)  # Ensure reasonable minimum premium
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def get_premium_insights(input_data, prediction):
    insights = []
    
    # Age-based insights
    age = input_data.get('Age', 35)
    if age < 25:
        insights.append("👶 <b>Young Driver</b>: Drivers under 25 typically have higher premiums due to less experience")
    elif age > 60:
        insights.append("👴 <b>Senior Driver</b>: Drivers over 60 may have adjusted premiums based on experience")
    else:
        insights.append("✅ <b>Prime Age</b>: You're in the preferred age range for insurance premiums")
    
    # Health-based insights
    health_score = input_data.get('Health Score', 75)
    if health_score < 40:
        insights.append("🏥 <b>Health Risk</b>: Lower health score may increase your premium. Consider improving health metrics")
    elif health_score > 80:
        insights.append("💪 <b>Excellent Health</b>: Your good health score is positively impacting your premium")
    
    # Smoking insights
    smoking_status = input_data.get('Smoking Status', 'No')
    if smoking_status == "Yes":
        insights.append("🚭 <b>Smoker</b>: Smoking status typically increases premiums due to higher health risks")
    else:
        insights.append("✅ <b>Non-Smoker</b>: Being a non-smoker helps keep your premium lower")
    
    # Income insights
    annual_income = input_data.get('Annual Income', 50000)
    if annual_income > 100000:
        insights.append("💰 <b>High Income</b>: Higher income may allow for more comprehensive coverage options")
    
    # Vehicle age insights
    vehicle_age = input_data.get('Vehicle Age', 5)
    if vehicle_age > 10:
        insights.append("🚗 <b>Older Vehicle</b>: Older vehicles may have different premium considerations")
    
    # Policy type insights
    policy_type = input_data.get('Policy Type', 'Comprehensive')
    if policy_type == "Premium":
        insights.append("🛡️ <b>Premium Coverage</b>: You've selected comprehensive protection")
    elif policy_type == "Basic":
        insights.append("📋 <b>Basic Coverage</b>: Consider upgrading for more protection")
    
    return insights

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 SmartPremium Insurance Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Predict Your Insurance Premium with Machine Learning")
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    
    if artifacts is None:
        st.stop()
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🎯 Premium Prediction", "📊 Model Insights"])
    
    with tab1:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("👤 Customer Information")
            
            # Personal details
            age = st.slider("**Age**", min_value=18, max_value=80, value=35, 
                           help="Age of the insured individual")
            
            gender = st.selectbox("**Gender**", ["Male", "Female", "Other"],
                                 help="Gender of the insured individual")
            
            annual_income = st.number_input("**Annual Income (₹)**", min_value=0, 
                                           value=50000, step=1000,
                                           help="Annual income in USD")
            
            health_score = st.slider("**Health Score**", min_value=0, max_value=100, 
                                    value=75, help="Health status score (0-100)")
            
            credit_score = st.slider("**Credit Score**", min_value=300, max_value=850, 
                                    value=700, help="Credit score (300-850)")
            
        with col2:
            st.header("📋 Policy & Lifestyle")
            
            # Policy details
            marital_status = st.selectbox("**Marital Status**", 
                                         ["Single", "Married", "Divorced", "Widowed"])
            
            num_dependents = st.number_input("**Number of Dependents**", min_value=0, 
                                            max_value=10, value=0)
            
            education = st.selectbox("**Education Level**", 
                                   ["High School", "Bachelor", "Master", "PhD", "Other"])
            
            location = st.selectbox("**Location Type**", 
                                  ["Urban", "Suburban", "Rural"])
            
            policy_type = st.selectbox("**Policy Type**", 
                                     ["Basic", "Comprehensive", "Premium"])
            
            vehicle_age = st.number_input("**Vehicle Age (years)**", min_value=0, 
                                         max_value=30, value=5)
            
            smoking_status = st.radio("**Smoking Status**", ["No", "Yes"])
            
            exercise_frequency = st.selectbox("**Exercise Frequency**", 
                                            ["Daily", "Weekly", "Monthly", "Rarely"])
        
        # Create input dictionary
        input_data = {
            'Age': age,
            'Gender': gender,
            'Annual Income': annual_income,
            'Health Score': health_score,
            'Credit Score': credit_score,
            'Marital Status': marital_status,
            'Number of Dependents': num_dependents,
            'Education Level': education,
            'Location': location,
            'Policy Type': policy_type,
            'Vehicle Age': vehicle_age,
            'Smoking Status': smoking_status,
            'Exercise Frequency': exercise_frequency
        }
        
        # Prediction button
        if st.button("🚀 Predict Insurance Premium", use_container_width=True, type="primary"):
            with st.spinner("🤖 Analyzing your profile and calculating premium..."):
                prediction = predict_premium(input_data, artifacts)
                
                if prediction is not None:
                    # Display prediction
                    # st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.success(f"## 💰 Predicted Insurance Premium: **₹{prediction:,.2f}**")
                        st.caption("per year")
                    
                    with col2:
                        monthly = prediction / 12
                        st.metric("Monthly Equivalent", f"₹{monthly:,.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show insights
                    st.subheader("📈 Premium Insights")
                    insights = get_premium_insights(input_data, prediction)
                    
                    for insight in insights:
                        st.markdown(f'<div class="insight-box">{insight}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Risk factors summary
                    st.subheader("🔍 Risk Factor Analysis")
                    
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    with risk_col1:
                        # Age risk
                        if age < 25 or age > 65:
                            st.error("**Age Risk**: Higher")
                        else:
                            st.success("**Age Risk**: Lower")
                    
                    with risk_col2:
                        # Health risk
                        if health_score < 50:
                            st.error("**Health Risk**: Higher")
                        else:
                            st.success("**Health Risk**: Lower")
                    
                    with risk_col3:
                        # Lifestyle risk
                        risk_factors = 0
                        if smoking_status == "Yes":
                            risk_factors += 1
                        if exercise_frequency == "Rarely":
                            risk_factors += 1
                        
                        if risk_factors >= 1:
                            st.warning(f"**Lifestyle Risk**: Moderate ({risk_factors} factors)")
                        else:
                            st.success("**Lifestyle Risk**: Lower")
    
    with tab2:
        st.header("🔍 Model Insights & Feature Importance")
        
        # Model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Algorithm**: {type(artifacts['model']).__name__}")
            st.write(f"**Features Used**: {len(artifacts['feature_columns'])}")
            st.write(f"**Training Data**: Insurance customer dataset")
            st.write(f"**Random State**: {artifacts['random_state']}")
        
        with col2:
            st.subheader("Model Performance")
            st.info("""
            This model was trained on historical insurance data and evaluates:
            - **RMSE** (Root Mean Square Error)
            - **MAE** (Mean Absolute Error) 
            - **R² Score** (Coefficient of Determination)
            """)
        
        # Feature importance
        if hasattr(artifacts['model'], 'feature_importances_'):
            st.subheader("📊 Top 10 Most Important Features")
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'Feature': artifacts['feature_columns'],
                'Importance': artifacts['model'].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            # Display as bars
            for _, row in feature_importance.iterrows():
                importance_pct = row['Importance'] * 100
                st.markdown(f"""
                <div class="feature-importance">
                    <strong>{row['Feature']}</strong>: {importance_pct:.1f}%
                    <div style="background: linear-gradient(90deg, #1f77b4 {importance_pct}%, #f0f0f0 {importance_pct}%); 
                                height: 20px; border-radius: 10px; margin-top: 5px;"></div>
                </div>
                """, unsafe_allow_html=True)
        
        # Data visualization section
        st.subheader("📈 Data Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            try:
                st.image('outputs/feature_importance.png', 
                        caption='Feature Importance Analysis', 
                        use_column_width=True)
            except:
                st.info("Feature importance visualization not available")
        
        with viz_col2:
            try:
                st.image('outputs/correlation_matrix.png', 
                        caption='Feature Correlation Matrix', 
                        use_column_width=True)
            except:
                st.info("Correlation matrix visualization not available")
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()