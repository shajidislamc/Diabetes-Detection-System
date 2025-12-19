import streamlit as st
import requests

API_BASE_URL = "https://diabetes-detection-system-npa2.onrender.com"
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes.")

# Input Form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    submit = st.form_submit_button("Predict")


# API Call
if submit:
    payload = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree_function": diabetes_pedigree_function,
        "age": age,
    }

    try:
        # 🔹 Wake up Render backend (cold start)
        requests.get(HEALTH_URL, timeout=60)

        with st.spinner("Predicting..."):
            response = requests.post(PREDICT_URL, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()

            st.subheader("🧪 Prediction Result")
            st.write(f"**Prediction:** {result['result']}")
            st.write(f"**Confidence:** {round(result['confidence'] * 100, 2)}%")

            if result["prediction"] == 1:
                st.error("⚠️ Diabetic")
            else:
                st.success("✅ Not Diabetic")

        else:
            st.error(f"❌ API Error (status {response.status_code})")
            st.code(response.text)

    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out. Backend may be waking up (Render cold start). Try again.")

    except requests.exceptions.ConnectionError:
        st.error("🚫 Cannot connect to backend API.")

    except Exception as e:
        st.error("Unexpected error occurred")
        st.code(str(e))
