


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

st.title("🔥 FitPredict")

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

    st.success(
        f"Estimated Calories Burned: {prediction[0]:.2f} kcal"
    )


st.divider()

st.caption(
    "Disclaimer: FitPredict provides an estimated calorie "
    "expenditure based on a machine learning model. "
    "It is not a medical device and should not be used "
    "for medical decisions."
)

