import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("✈️ Customer Churn Prediction")
st.markdown("### Predict whether a travel customer will churn using a Random Forest model")
st.markdown("---")

st.sidebar.header("📋 Customer Details")

age = st.sidebar.slider("Age", min_value=18, max_value=80, value=35, step=1)

frequent_flyer = st.sidebar.selectbox(
    "Frequent Flyer?",
    options=["No", "Yes"],
    index=0
)

annual_income_class = st.sidebar.selectbox(
    "Annual Income Class",
    options=["Low Income", "Middle Income", "High Income"],
    index=1
)

services_opted = st.sidebar.slider(
    "Number of Services Opted", min_value=1, max_value=9, value=4, step=1
)

account_synced = st.sidebar.selectbox(
    "Account Synced to Social Media?",
    options=["No", "Yes"],
    index=0
)

booked_hotel = st.sidebar.selectbox(
    "Booked Hotel or Not?",
    options=["No", "Yes"],
    index=0
)

def encode_input(age, frequent_flyer, annual_income_class, services_opted, account_synced, booked_hotel):
    ff_map = {"No": 0, "Yes": 1}
    aic_map = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
    yn_map = {"No": 0, "Yes": 1}
    return np.array([[
        age,
        ff_map[frequent_flyer],
        aic_map[annual_income_class],
        services_opted,
        yn_map[account_synced],
        yn_map[booked_hotel]
    ]])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🧾 Customer Input Summary")
    input_df = pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                    "Services Opted", "Account Synced to Social Media", "Booked Hotel"],
        "Value": [age, frequent_flyer, annual_income_class,
                  services_opted, account_synced, booked_hotel]
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Prediction Result")
    if predict_btn:
        input_array = encode_input(age, frequent_flyer, annual_income_class,
                                   services_opted, account_synced, booked_hotel)
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]

        if prediction == 1:
            st.error("⚠️ **HIGH CHURN RISK** — This customer is likely to churn!")
        else:
            st.success("✅ **LOW CHURN RISK** — This customer is likely to stay!")

        st.markdown(f"**Confidence:** {max(probability)*100:.1f}%")

        churn_prob = probability[1]
        st.markdown(f"**Churn Probability:** `{churn_prob*100:.1f}%`")
        st.progress(float(churn_prob))

        not_churn_prob = probability[0]
        st.markdown(f"**Retention Probability:** `{not_churn_prob*100:.1f}%`")

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(
            ["Not Churned", "Churned"],
            [not_churn_prob, churn_prob],
            color=["steelblue", "tomato"],
            width=0.4
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability", fontweight="bold")
        for bar, val in zip(bars, [not_churn_prob, churn_prob]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.2%}", ha="center", fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("👈 Fill in customer details and click **Predict Churn**")

st.markdown("---")
st.subheader("🔍 Feature Importance")

feature_names = ["Age", "FrequentFlyer", "AnnualIncomeClass",
                 "ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

fig2, ax2 = plt.subplots(figsize=(8, 4))
bars = ax2.barh(feat_df["Feature"], feat_df["Importance"], color="teal")
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance – Random Forest", fontweight="bold")
for bar, val in zip(bars, feat_df["Importance"]):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center")
st.pyplot(fig2)

st.markdown("---")
st.caption("🎓 B.Tech Gen AI – Final Project | Customer Churn Prediction using Random Forest")
