import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard predicts the likelihood of customer churn based on various features.
Fill out the customer profile below and get an AI-driven prediction powered by our deployed MLOps backend.
""")

# Logical groupings
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen_str = st.radio("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen_str == "Yes" else 0
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.header("💳 Account Details")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

with col3:
    st.header("🌐 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown("---")
# API URL (update this after deployment or if running locally)
API_URL = "https://customer-churn-prediction-hrqz.onrender.com"

st.subheader("🚀 Run Prediction")
predict_button = st.button("Predict Churn Risk", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Making prediction over the network..."):
        # Match strict Pydantic schema keys and types
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=customer_data)
            response.raise_for_status()
            result = response.json()
            
            # Display results beautifully
            st.markdown("### 📊 Prediction Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
            with res_col2:
                st.metric("Prediction", result['churn_prediction'])
            with res_col3:
                st.metric("Confidence", result['confidence'])
                
            # Visual indicator mapping
            if result['churn_prediction'] == "Yes":
                st.error("🚨 **High Risk:** This customer is likely to churn!")
                st.warning("""
                **Recommendation:**
                - Offer retention discount
                - Provide premium support
                - Review contract terms
                - Conduct satisfaction survey
                """)
            else:
                st.success("✅ **Low Risk:** This customer is likely to stay.")
                st.info("""
                **Recommendation:**
                - Maintain current service level
                - Consider upsell opportunities
                - Monitor for changes in behavior
                """)
                
        except requests.exceptions.RequestException as e:
            st.error("❌ **Network Error:** Could not reach the API. Is the backend container running?")
            st.exception(e)
        except json.JSONDecodeError as e:
            st.error("❌ **Parsing Error:** Received invalid JSON from the API.")
            st.exception(e)
        except Exception as e:
            st.error("❌ **Unexpected Error:** Something went wrong.")
            st.exception(e)

# Add health check check to verify backend status
st.sidebar.markdown("### ⚙️ Backend Status")
try:
    health_res = requests.get(f"{API_URL}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend: Online ✅")
    else:
        st.sidebar.warning(f"Backend: Warning {health_res.status_code}")
except:
    st.sidebar.error("Backend: Offline ❌")

