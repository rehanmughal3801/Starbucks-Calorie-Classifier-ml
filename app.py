import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Starbucks Calorie Predictor", page_icon="☕", layout="centered")

MODEL_PATH = Path(__file__).with_name("starbucks_decision_tree_model.joblib")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)

st.title("☕ Starbucks Calorie Classifier")
st.write("Decision Tree model se drink ki calorie category predict karein.")
try:
    bundle = load_model()
    model = bundle["model"]
    imputer = bundle["imputer"]
    features = bundle["features"]
except Exception as exc:
    st.error(f"Model load nahi ho saka: {exc}")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.metric("Test accuracy", f'{bundle["accuracy"] * 100:.1f}%')
    st.caption("DecisionTreeClassifier • max_depth=5")

st.subheader("Nutrition details enter karein")
defaults = {
    "Total Fat (g)": 0.0, "Trans Fat (g)": 0.0, "Saturated Fat (g)": 0.0,
    "Sodium (mg)": 10.0, "Total Carbohydrates (g)": 30.0,
    "Cholesterol (mg)": 0.0, "Dietary Fibre (g)": 0.0, "Sugars (g)": 25.0,
    "Protein (g)": 2.0, "Vitamin A (% DV)": 0.0, "Vitamin C (% DV)": 0.0,
    "Calcium (% DV)": 0.0, "Iron (% DV)": 0.0, "Caffeine (mg)": 100.0
}
with st.form("nutrition_form"):
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            values[feature] = st.number_input(
                feature, min_value=0.0, max_value=10000.0,
                value=float(defaults.get(feature, 0.0)), step=1.0
            )
    submitted = st.form_submit_button("Predict calorie category", type="primary")

if submitted:
    input_df = pd.DataFrame([[values[f] for f in features]], columns=features)
    transformed = imputer.transform(input_df)
    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]
    st.success(f"Predicted category: **{prediction}**")
    st.subheader("Prediction confidence")
    confidence = float(max(probabilities))
    st.progress(confidence)
    st.write(f"{confidence * 100:.1f}%")
    st.caption("Note: This model predicts a calorie category, not an exact calorie count.")
