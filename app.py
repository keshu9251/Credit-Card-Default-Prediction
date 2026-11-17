
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="CreditGuard | Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("credit_card_default_model.pkl")

model = load_model()

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- FEATURE NAMES ----------------
feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
    'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172554;
}
.subtitle {
    color: #64748B;
    font-size: 15px;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">💳 CreditGuard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Intelligent Credit Card Default Prediction System</div>',
    unsafe_allow_html=True
)
st.caption(
    "Project by: Keshav Agarwal (Roll No. 253301024) | "
    "Anirudh Kalla (Roll No. 253301005) | MCA 3rd Semester"
)

st.write("")
st.divider()

# ---------------- DASHBOARD METRICS ----------------
st.subheader("Model Performance Dashboard")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Best Model", "Tuned Random Forest")
m2.metric("Accuracy", "78.09%")
m3.metric("Precision", "50.45%")
m4.metric("ROC-AUC", "76.79%")

st.caption(
    "Metrics are from the project's held-out test-set evaluation."
)

st.divider()

# ---------------- INPUT FORM ----------------
st.subheader("🔍 Customer Risk Assessment")
st.write("Enter customer details to generate a prediction.")

with st.form("prediction_form"):

    st.markdown("#### 1. Personal and Credit Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        limit_bal = st.number_input(
            "Credit Limit",
            min_value=0,
            value=50000,
            step=10000
        )
        sex = st.selectbox(
            "Gender",
            [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

    with c2:
        education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Others",
                5: "Unknown",
                6: "Unknown"
            }[x]
        )
        marriage = st.selectbox(
            "Marital Status",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Married",
                2: "Single",
                3: "Others"
            }[x]
        )

    with c3:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    st.divider()
    st.markdown("#### 2. Repayment Status")

    pay_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    p1, p2, p3 = st.columns(3)

    with p1:
        pay_0 = st.selectbox("Last Month (PAY_0)", pay_options, index=2)
        pay_2 = st.selectbox("2 Months Ago (PAY_2)", pay_options, index=2)

    with p2:
        pay_3 = st.selectbox("3 Months Ago (PAY_3)", pay_options, index=2)
        pay_4 = st.selectbox("4 Months Ago (PAY_4)", pay_options, index=2)

    with p3:
        pay_5 = st.selectbox("5 Months Ago (PAY_5)", pay_options, index=2)
        pay_6 = st.selectbox("6 Months Ago (PAY_6)", pay_options, index=2)

    st.divider()
    st.markdown("#### 3. Previous Bill Amounts")

    b1, b2, b3 = st.columns(3)

    with b1:
        bill_amt1 = st.number_input("Bill Amount 1", value=5000)
        bill_amt2 = st.number_input("Bill Amount 2", value=5000)

    with b2:
        bill_amt3 = st.number_input("Bill Amount 3", value=5000)
        bill_amt4 = st.number_input("Bill Amount 4", value=5000)

    with b3:
        bill_amt5 = st.number_input("Bill Amount 5", value=5000)
        bill_amt6 = st.number_input("Bill Amount 6", value=5000)

    st.divider()
    st.markdown("#### 4. Previous Payment Amounts")

    a1, a2, a3 = st.columns(3)

    with a1:
        pay_amt1 = st.number_input("Payment Amount 1", min_value=0, value=1000)
        pay_amt2 = st.number_input("Payment Amount 2", min_value=0, value=1000)

    with a2:
        pay_amt3 = st.number_input("Payment Amount 3", min_value=0, value=1000)
        pay_amt4 = st.number_input("Payment Amount 4", min_value=0, value=1000)

    with a3:
        pay_amt5 = st.number_input("Payment Amount 5", min_value=0, value=1000)
        pay_amt6 = st.number_input("Payment Amount 6", min_value=0, value=1000)

    submitted = st.form_submit_button(
        "🚀 Predict Customer Default",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:

    customer_data = [[
        limit_bal, sex, education, marriage, age,
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
        bill_amt1, bill_amt2, bill_amt3,
        bill_amt4, bill_amt5, bill_amt6,
        pay_amt1, pay_amt2, pay_amt3,
        pay_amt4, pay_amt5, pay_amt6
    ]]

    input_df = pd.DataFrame(
        customer_data,
        columns=feature_names
    )

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1] * 100)

    result = (
        "May Default" if prediction == 1
        else "May Not Default"
    )

    st.session_state.history.append({
        "Customer": f"Customer {len(st.session_state.history) + 1}",
        "Prediction": result,
        "Default Probability": round(probability, 2)
    })

    st.divider()
    st.subheader("📊 Prediction Result")

    r1, r2 = st.columns(2)

    with r1:
        if prediction == 1:
            st.error("⚠️ Customer may default")
        else:
            st.success("✅ Customer may not default")

    with r2:
        st.metric(
            "Estimated Default Probability",
            f"{probability:.2f}%"
        )

    probability_chart = pd.DataFrame({
        "Category": ["Default", "No Default"],
        "Probability": [probability, 100 - probability]
    })

    fig = px.bar(
        probability_chart,
        x="Category",
        y="Probability",
        color="Category",
        text_auto=".2f",
        title="Prediction Probability"
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The model uses a 50% probability threshold. "
        "This prediction is an estimate, not a guarantee."
    )

# ---------------- PREDICTION HISTORY ----------------
st.divider()
st.subheader("📁 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions made yet. Submit customer details to begin.")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "CreditGuard | MCA Machine Learning Project | "
    "Credit Card Default Prediction"
)
