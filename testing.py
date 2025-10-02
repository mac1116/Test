import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and expected feature order
model, scaler, features = joblib.load("personality_model_and_scaler.pkl")

st.set_page_config(page_title="Personality Prediction", layout="centered")
st.title("🧠 Personality Prediction App")
st.write("Choose a mode below, then provide input or upload a CSV for batch prediction.")

# --- Persist selected mode in session_state ---
if "mode" not in st.session_state:
    st.session_state.mode = "Manual Input"   # default

# --- Presentable top buttons (store selection) ---
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("✍️ Manual Input", key="btn_manual"):
        st.session_state.mode = "Manual Input"
with col2:
    if st.button("📂 CSV Upload", key="btn_csv"):
        st.session_state.mode = "CSV Upload"
with col3:
    if st.button("🔁 Reset", key="btn_reset"):
        st.session_state.mode = "Manual Input"

st.markdown("---")
st.info(f"**Selected mode:** {st.session_state.mode}")

# --- MANUAL MODE ---
if st.session_state.mode == "Manual Input":
    st.subheader("✍️ Manual Input")
    time_spent_alone = st.number_input("Time spent alone (hours)", 0, 24, 5)
    social_event_attendance = st.slider("Social event attendance (per month)", 0, 30, 3)
    going_outside = st.slider("Going outside (days per week)", 0, 7, 2)
    friends_circle_size = st.number_input("Friends circle size", 0, 100, 10)
    post_frequency = st.slider("Post frequency (per week)", 0, 50, 1)
    stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"], horizontal=True)
    drained_after_socializing = st.radio("Do you feel drained after socializing?", ["Yes", "No"], horizontal=True)

    # encode
    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained_after_socializing == "Yes" else 0

    input_data = pd.DataFrame([{
        "Time_spent_Alone": time_spent_alone,
        "Social_event_attendance": social_event_attendance,
        "Going_outside": going_outside,
        "Friends_circle_size": friends_circle_size,
        "Post_frequency": post_frequency,
        "Stage_fear": stage_fear_val,
        "Drained_after_socializing": drained_val
    }])

    # Ensure correct column order
    input_data = input_data[features]

    if st.button("🔮 Predict Personality", key="predict_manual"):
        try:
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            personality = "Extrovert 🎉" if pred == 1 else "Introvert 🌱"
            st.success(f"Predicted Personality: **{personality}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- CSV UPLOAD MODE ---
elif st.session_state.mode == "CSV Upload":
    st.subheader("📂 CSV Upload for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Convert Yes/No to 1/0 where applicable
        for col in ["Stage_fear", "Drained_after_socializing"]:
            if col in df_uploaded.columns:
                df_uploaded[col] = df_uploaded[col].replace(
                    {"Yes": 1, "No": 0, "yes": 1, "no": 0, "YES": 1, "NO": 0}
                )

        # Check required columns
        missing_cols = [col for col in features if col not in df_uploaded.columns]
        if missing_cols:
            st.error(f"The uploaded file is missing these columns: {missing_cols}")
        else:
            st.write("✅ File preview:")
            st.dataframe(df_uploaded.head())

            if st.button("🔮 Predict All (CSV)", key="predict_csv"):
                try:
                    X = df_uploaded[features].astype(float)
                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)
                    df_uploaded["Predicted_Personality"] = ["Extrovert 🎉" if p == 1 else "Introvert 🌱" for p in preds]
                    st.success("Batch prediction completed!")
                    st.dataframe(df_uploaded)

                    csv_output = df_uploaded.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions as CSV", data=csv_output,
                                       file_name="predicted_personality.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 | Created by **Mheil Andrei Cenita**, **Yosh B. Batula**, **Kent Sevellino**")
