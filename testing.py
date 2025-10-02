import streamlit as st
import pandas as pd
import joblib

model, scaler, features = joblib.load("personality_model_and_scaler.pkl")


st.title("🧠 Personality Prediction App")
st.write("Welcome to the testing page! Please fill in the details below:")


time_spent_alone = st.number_input("Time spent alone (hours)", 0, 24, 5)
social_event_attendance = st.slider("Social event attendance (per month)", 0, 30, 3)
going_outside = st.slider("Going outside (days per week)", 0, 7, 2)
friends_circle_size = st.number_input("Friends circle size", 0, 100, 10)
post_frequency = st.slider("Post frequency (per week)", 0, 50, 1)

stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])
drained_after_socializing = st.radio("Do you feel drained after socializing?", ["Yes", "No"])


stage_fear = 1 if stage_fear == "Yes" else 0    
drained_after_socializing = 1 if drained_after_socializing == "Yes" else 0


input_data = pd.DataFrame([{
    "Time_spent_Alone": time_spent_alone,
    "Social_event_attendance": social_event_attendance,
    "Going_outside": going_outside,
    "Friends_circle_size": friends_circle_size,
    "Post_frequency": post_frequency,
    "Stage_fear": stage_fear,
    "Drained_after_socializing": drained_after_socializing
}])


input_scaled = scaler.transform(input_data[features])

if st.button("Predict Personality"):
    prediction = model.predict(input_scaled)[0]
    personality = "Extrovert 🎉" if prediction == 1 else "Introvert 🌱"
    st.success(f"Predicted Personality: **{personality}**")

st.markdown(
    """
    ---
    © 2025 | Created by **Mheil Andrei Cenita**, **Yosh B. Batula**, **Kent Sevellino**
    """
)
