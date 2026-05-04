import streamlit as st
import joblib

# Load model + encoder
model = joblib.load("outputs/spam_detector_model.pkl")
label_encoder = joblib.load("outputs/label_encoder.pkl")

st.title("📧 Spam Detection Web App")

# Input box
text = st.text_area("Enter email/message")

# Button
if st.button("Check"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0].max()
        label = label_encoder.inverse_transform([pred])[0]

        if label.lower() == "spam":
            st.error(f"SPAM ({prob*100:.2f}%)")
        else:
            st.success(f"HAM ({prob*100:.2f}%)")