


import streamlit as st
import pandas as pd
import joblib
import os




BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODELS_PATH = os.path.join(BASE_DIR, "Models")


model = joblib.load(
    os.path.join(MODELS_PATH, "random_forest_model.pkl")
)


encoder = joblib.load(
    os.path.join(MODELS_PATH, "gender_encoder.pkl")
)


model_features = joblib.load(
    os.path.join(MODELS_PATH, "model_features.pkl")
)



st.set_page_config(
    page_title="FitPredict",
    page_icon="🔥",
    layout="centered"
)

# ==========================================
# CUSTOM FITPREDICT STYLING
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #0B1220;
    color: #F8FAFC;
}

/* Main content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Main heading */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #A3E635;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section headings */
h2, h3 {
    color: #A3E635 !important;
}

/* Input labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #E2E8F0 !important;
    font-weight: 600;
}

/* Input fields */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #1E293B;
    color: white;
}

/* Buttons */
.stButton > button {
    background-color: #84CC16;
    color: #0B1220;
    font-weight: 700;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    width: 100%;
}

/* Button hover */
.stButton > button:hover {
    background-color: #A3E635;
    color: #0B1220;
    border: none;
}

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, #1E293B, #172554);
    border: 1px solid #84CC16;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}

/* Result number */
.result-value {
    font-size: 48px;
    font-weight: 800;
    color: #A3E635;
}

/* Result description */
.result-label {
    color: #CBD5E1;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)





st.markdown(
    '<div class="main-title">🔥 FitPredict</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Intelligent Exercise Calorie Estimator'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Estimate exercise-related calorie expenditure "
    "using machine learning and physiological data."
)


st.subheader(
    "Machine Learning-Based Calorie Expenditure Prediction"
)

st.write(
    "Enter your exercise details below to estimate "
    "your calorie expenditure using Machine Learning."
)

st.divider()


st.header("Enter Your Exercise Details")

gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

age = st.number_input(
    "Age (years)",
    min_value=20,
    max_value=79,
    value=27
)

height = st.number_input(
    "Height (cm)",
    min_value=123.0,
    max_value=222.0,
    value=170.0
)

weight = st.number_input(
    "Weight (kg)",
    min_value=36.0,
    max_value=132.0,
    value=65.0
)

duration = st.number_input(
    "Exercise Duration (minutes)",
    min_value=1.0,
    max_value=30.0,
    value=10.0
)

heart_rate = st.number_input(
    "Heart Rate (bpm)",
    min_value=67.0,
    max_value=128.0,
    value=90.0
)

body_temp = st.number_input(
    "Body Temperature (°C)",
    min_value=37.1,
    max_value=41.5,
    value=39.0
)



def prepare_input(gender, age, height, weight, duration, heart_rate, body_temp):

    input_data = pd.DataFrame({
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp]
    })

    gender_data = pd.DataFrame({
        "Gender": [gender]
    })

    gender_encoded = encoder.transform(gender_data)

    gender_encoded_df = pd.DataFrame(
        gender_encoded,
        columns=encoder.get_feature_names_out(["Gender"])
    )

    final_input = pd.concat(
        [input_data, gender_encoded_df],
        axis=1
    )

    final_input = final_input.reindex(
        columns=model_features
    )

    return final_input



if st.button("Predict Calories Burned"):

    final_input = prepare_input(
        gender,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    )

    prediction = model.predict(final_input)

   
if st.button("🔥 Predict Calories Burned"):

    final_input = prepare_input(
        gender,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    )

    prediction = model.predict(final_input)[0]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                YOUR ESTIMATED CALORIE EXPENDITURE
            </div>

            <div class="result-value">
                {prediction:.2f} kcal
            </div>

            <div class="result-label">
                Prediction generated successfully
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Your Exercise Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Duration", f"{duration} min")
    col2.metric("Heart Rate", f"{heart_rate} bpm")
    col3.metric("Body Temperature", f"{body_temp} °C")



st.divider()

st.caption(
    "Disclaimer: FitPredict provides an estimated calorie "
    "expenditure based on a machine learning model. "
    "It is not a medical device and should not be used "
    "for medical decisions."
)

