
import streamlit as st
import pandas as pd
import joblib
import base64

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #061629, #102F50);
    color: white;
}

.main-title {
    text-align: center;
    color: #38BDF8 !important;
    font-size: 48px;
    font-weight: 900;
}

.subtitle {
    text-align: center;
    color: #FFFFFF !important;
    font-size: 20px;
}

.welcome-text {
    text-align: center;
    color: #FFFFFF !important;
    font-size: 19px;
    line-height: 1.8;
}

.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp .stMarkdown p {
    color: #FFFFFF !important;
}

.stNumberInput input {
    background: white !important;
    color: #102F50 !important;
}

.stButton > button {
    background: #38BDF8 !important;
    color: #061629 !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: bold !important;
    min-height: 48px;
}

.stButton > button p {
    color: #061629 !important;
}

.stButton > button:hover {
    background: #7DD3FC !important;
}

div[data-testid="stMetric"] {
    background: #163653;
    padding: 15px;
    border-radius: 12px;
}

hr {
    border-color: #38BDF8 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "page" not in st.session_state:
    st.session_state.page = "welcome"


# ---------------- WELCOME PAGE ----------------

if st.session_state.page == "welcome":

    st.write("")
    st.markdown(
        '<div class="main-title">💧 AQUAGUARD AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'SMART WATER LEAKAGE PREDICTION SYSTEM'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # Display animated GIF
    try:
        with open("app/aquaguard_welcome.gif", "rb") as file:
            gif_data = base64.b64encode(file.read()).decode()

        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:image/gif;base64,{gif_data}"
                style="width:100%; max-width:850px;
                border-radius:20px;
                box-shadow:0 0 25px #0284C7;">
            </div>
            """,
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.warning(
            "Welcome animation not found. "
            "Please place aquaguard_welcome.gif inside app folder."
        )

    st.write("")

    st.markdown(
        """
        <div class="welcome-text">
            <h2>Welcome to AquaGuard AI!</h2>
            An intelligent machine learning system designed
            to predict water leakage using pipe sensor data.
            <br><br>
            💧 Smart Monitoring &nbsp; | &nbsp;
            🤖 Machine Learning &nbsp; | &nbsp;
            🌍 Save Water
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, center, right = st.columns([1, 2, 1])

    with center:
        if st.button("🚀 GET STARTED", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

    st.write("")
    st.caption("AquaGuard AI | Machine Learning Project")


# ---------------- DASHBOARD PAGE ----------------

elif st.session_state.page == "dashboard":

    st.markdown(
        '<div class="main-title">💧 AQUAGUARD AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Water Leakage Prediction Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    if st.button("⬅ BACK TO WELCOME"):
        st.session_state.page = "welcome"
        st.rerun()

    st.subheader("⚙️ ENTER PIPE SENSOR DETAILS")

    col1, col2 = st.columns(2)

    with col1:
        pressure = st.number_input(
            "Pressure (PSI)", min_value=0.0, value=50.0
        )
        flow = st.number_input(
            "Flow Rate (L/min)", min_value=0.0, value=30.0
        )
        temperature = st.number_input(
            "Temperature (°C)", value=25.0
        )

    with col2:
        vibration = st.number_input(
            "Vibration (mm/s)", min_value=0.0, value=2.0
        )
        rpm = st.number_input(
            "RPM", min_value=0.0, value=1000.0
        )
        hours = st.number_input(
            "Operational Hours", min_value=0.0, value=500.0
        )

    st.write("")

    if st.button(
        "🔍 PREDICT WATER LEAKAGE",
        use_container_width=True
    ):

        try:
            model = joblib.load("models/water_leakage_model.pkl")

            input_data = pd.DataFrame([[
                pressure, flow, temperature,
                vibration, rpm, hours
            ]], columns=[
                "Pressure_PSI",
                "Flow_Rate_Lmin",
                "Temperature_C",
                "Vibration_mms",
                "RPM",
                "Operational_Hours_h"
            ])

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            confidence = probabilities[
                list(model.classes_).index(prediction)
            ] * 100

            st.divider()
            st.subheader("📊 PREDICTION RESULT")

            if prediction == 1:
                st.error("🚨 WATER LEAKAGE DETECTED!")
            else:
                st.success("✅ NO WATER LEAKAGE DETECTED")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(float(confidence / 100))

            st.caption(
                "Model confidence is an estimate, "
                "not a guarantee of real-world accuracy."
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    st.divider()
    st.caption("💧 AquaGuard AI | Machine Learning Project")
