import os
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ---------- THEME ----------
st.set_page_config(page_title="🧠 Stroke Risk Predictor", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .stButton>button {
            background-color: #4e79a7;
            color: white;
            font-weight: bold;
        }
        .stProgress > div > div {
            background-color: #4e79a7;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_models():
    try:
        clf = joblib.load("models/stroke_risk_classifier.pkl")
        reg = joblib.load("models/stroke_risk_regressor.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return clf, reg, scaler
    except:
        st.error("❌ Models not found.")
        return None, None, None

@st.cache_data
def load_feature_names():
    return [
        'Age', 'Chest Pain', 'Shortness of Breath', 'Irregular Heartbeat', 'Fatigue & Weakness',
        'Dizziness', 'Swelling (Edema)', 'Pain in Neck/Jaw/Shoulder/Back', 'Excessive Sweating',
        'Persistent Cough', 'Nausea/Vomiting', 'High Blood Pressure', 'Chest Discomfort (Activity)',
        'Cold Hands/Feet', 'Snoring/Sleep Apnea', 'Anxiety/Feeling of Doom'
    ]

# ---------- SIMPLE EXPLAINABILITY ----------
def get_top_features(input_data):
    sorted_feats = sorted(input_data.items(), key=lambda x: x[1], reverse=True)
    return sorted_feats[:3]

# ---------- MAIN APP ----------
def main():
    st.title("🧠 Stroke Risk Prediction")
    st.subheader("Predict stroke risk based on patient health indicators.")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        - 🔍 Predict stroke **risk percentage**
        - ⚠️ Identify patients **at risk**
        - 📄 Download a report
        - 📈 See top contributing symptoms
        """)
    
    clf, reg, scaler = load_models()
    feature_names = load_feature_names()
    if not all([clf, reg, scaler]):
        return

    with st.form("prediction_form"):
        st.header("👤 Patient Info")

        age = st.slider("Age", 0, 120, 50)
        col1, col2 = st.columns(2)
        symptom_inputs = {}

        for i, feature in enumerate(feature_names[1:]):
            with (col1 if i % 2 == 0 else col2):
                symptom_inputs[feature] = st.selectbox(f"{feature}", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        submitted = st.form_submit_button("🚀 Predict Stroke Risk")
    
    if submitted:
        input_data = {'Age': age}
        input_data.update(symptom_inputs)
        input_array = [input_data[feature] for feature in feature_names]
        input_scaled = scaler.transform([input_array])

        at_risk = clf.predict(input_scaled)[0]
        stroke_risk = reg.predict(input_scaled)[0]

        # ---------- RESULTS ----------
        st.success("✅ Prediction complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("At Risk", "Yes" if at_risk else "No", "⚠️" if at_risk else "✅")
        with col2:
            st.metric("Stroke Risk (%)", f"{stroke_risk:.1f}%", 
                      "🔴 High" if stroke_risk > 50 else "🟠 Moderate" if stroke_risk > 25 else "🟢 Low")
        st.progress(min(int(stroke_risk), 100) / 100)

        # ---------- EXPLAINABILITY ----------
        st.subheader("🔍 Key Contributing Factors")
        top_feats = get_top_features(input_data)
        st.write("Top symptoms influencing this prediction:")
        for feat, value in top_feats:
            st.write(f"- **{feat}**: {'Yes' if value else 'No'}")

        # ---------- CHART ----------
        st.subheader("📊 Risk Breakdown Chart")
        fig, ax = plt.subplots()
        ax.bar(["Stroke Risk", "Remaining"], [stroke_risk, 100-stroke_risk], color=["#e15759", "#bab0ab"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Predicted Stroke Risk")
        st.pyplot(fig)

        # ---------- REPORT DOWNLOAD ----------
        st.subheader("📄 Download Report")
        report_content = f"""Stroke Prediction Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

Age: {age}
Predicted Risk: {stroke_risk:.1f}%
At Risk: {"Yes" if at_risk else "No"}

Top Contributing Factors:
"""
        for feat, value in top_feats:
            report_content += f"- {feat}: {'Yes' if value else 'No'}\n"

        st.download_button("💾 Download Report", report_content, file_name="stroke_risk_report.txt")

if __name__ == "__main__":
    main()
