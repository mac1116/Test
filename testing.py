import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model, scaler, features = joblib.load("personality_model_and_scaler.pkl")

st.title("🧠 Personality Prediction App")
st.write("Welcome! You can either fill in the details manually OR upload a CSV file for batch prediction.")

# --- Manual Input Section ---
st.subheader("📌 Manual Input")

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

if st.button("Predict Personality (Manual Input)"):
    prediction = model.predict(input_scaled)[0]
    personality = "Extrovert 🎉" if prediction == 1 else "Introvert 🌱"
    st.success(f"Predicted Personality: **{personality}**")

# --- CSV Upload Section ---
st.subheader("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    # Encode categorical columns (same as training)
    if "Stage_fear" in df_uploaded.columns:
        df_uploaded["Stage_fear"] = df_uploaded["Stage_fear"].replace({"Yes": 1, "No": 0}).astype(int)

    if "Drained_after_socializing" in df_uploaded.columns:
        df_uploaded["Drained_after_socializing"] = df_uploaded["Drained_after_socializing"].replace({"Yes": 1, "No": 0}).astype(int)

    # Check if all required columns are present
    missing_cols = [col for col in features if col not in df_uploaded.columns]
    if missing_cols:
        st.error(f"The uploaded file is missing these columns: {missing_cols}")
    else:
        st.write("✅ File successfully uploaded! Here’s a preview:")
        st.dataframe(df_uploaded.head())

        # Scale and predict
        uploaded_scaled = scaler.transform(df_uploaded[features])
        predictions = model.predict(uploaded_scaled)

        df_uploaded["Predicted_Personality"] = ["Extrovert 🎉" if p == 1 else "Introvert 🌱" for p in predictions]

        st.write("📊 Prediction Results:")
        st.dataframe(df_uploaded)

        # Option to download results
        csv_output = df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv_output, file_name="predicted_personality.csv", mime="text/csv")


st.markdown(
    """
    ---
    © 2025 | Created by **Mheil Andrei Cenita**, **Yosh B. Batula**, **Kent Sevellino**
    """
)
