import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="College Psychiatric Referral Predictor", layout="wide")


@st.cache_resource
def load_models():
    scaler = joblib.load("scaler_top15.pkl")
    model_calib = joblib.load("xgb_model_top15_calibrated.pkl")
    model_uncalib = joblib.load("xgb_model_top15_uncalibrated.pkl")
    features = joblib.load("top15_features_list.pkl")
    threshold = float(joblib.load("optimal_threshold.pkl"))
    return scaler, model_calib, model_uncalib, features, threshold


try:
    scaler, model_calib, model_uncalib, features, threshold = load_models()
except Exception:
    st.error("Model components not found. Please ensure all 5 .pkl files are in the directory.")
    st.stop()


st.markdown(
    "<h1 style='text-align: center;'>College Psychiatric Referral Predictor</h1>",
    unsafe_allow_html=True,
)
st.write("")


scl_dict = {
    "SCL_2": "Nervousness",
    "SCL_5": "Loss of sexual interest",
    "SCL_14": "Low energy",
    "SCL_15": "Suicidal thoughts",
    "SCL_20": "Crying easily",
    "SCL_21": "Uneasy with opposite sex",
    "SCL_24": "Temper outbursts",
    "SCL_26": "Self-blame",
    "SCL_28": "Difficulty completing tasks",
    "SCL_32": "Loss of interest",
    "SCL_36": "Feeling misunderstood",
    "SCL_44": "Trouble falling asleep",
    "SCL_69": "Self-conscious with others",
    "SCL_78": "Restlessness",
    "SCL_82": "Fear of fainting in public",
    "SCL_90": "Feeling mind is wrong",
}


cols = st.columns(5)
user_inputs = {}

for i, feature in enumerate(features):
    with cols[i % 5]:
        desc = scl_dict.get(feature, "")
        display_label = f"{desc} ({feature})" if desc else feature

        user_inputs[feature] = st.number_input(
            label=display_label,
            min_value=1,
            max_value=5,
            value=1,
            step=1,
        )

st.write("")

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict_btn = st.button("Calculate score", use_container_width=True)


if predict_btn:
    input_df = pd.DataFrame([user_inputs], columns=features)
    input_scaled = scaler.transform(input_df)

    risk_prob = model_calib.predict_proba(input_scaled)[0][1]

    explainer = shap.TreeExplainer(model_uncalib)
    shap_values = explainer.shap_values(input_scaled)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    shap_val = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_val = shap_val[0]

    st.markdown("---")

    st.markdown(
        (
            "<div style='text-align: center; font-size: 50px; font-weight: 900; "
            "color: #1f77b4; margin-bottom: 20px;'>"
            "The model-estimated likelihood of belonging to the clinically diagnosed "
            f"reference cohort is {risk_prob * 100:.1f}%."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    plt.clf()
    plt.rcParams.update({"font.size": 8})

    shap.force_plot(
        base_value,
        shap_val,
        input_df.iloc[0],
        feature_names=features,
        out_names="Cohort similarity score",
        matplotlib=True,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(20, 3)
    st.pyplot(fig, bbox_inches="tight", use_container_width=True)

    st.write("")

    threshold_pct = threshold * 100
    st.markdown(
        (
            "<div style='text-align: center; font-size: 36px; font-weight: 900; "
            "color: #2e7d32; margin-top: 20px;'>"
            f"Higher-priority threshold: {threshold_pct:.1f}%. "
            f"Scores &ge;{threshold_pct:.1f}% suggest greater similarity to the "
            "clinically diagnosed reference cohort and may warrant professional "
            "mental health assessment."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            "<p style='text-align: center; font-size: 14px; color: gray; margin-top: 30px;'>"
            "This research prototype supports post-screening risk stratification only. "
            "It is not a diagnostic tool, does not determine whether clinical intervention "
            "is required, and cannot replace face-to-face assessment by a qualified mental "
            "health professional."
            "</p>"
        ),
        unsafe_allow_html=True,
    )
