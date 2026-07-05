import streamlit as st
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Myth Buster",
    page_icon="🛡️",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------
# Title
# -------------------------
st.title("🛡️ Myth Buster")
st.subheader("Fake / Fraud Message Detection")

st.write(
    "Paste an SMS or message below and the model will predict whether it is **Legitimate** or **Fraud/Spam**."
)

st.divider()

# -------------------------
# Text Input
# -------------------------
message = st.text_area(
    "Enter your message",
    height=180,
    placeholder="Paste your message here..."
)

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        vector = vectorizer.transform([message])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = probability[prediction] * 100

        st.divider()

        st.subheader("Prediction")

        if prediction == 1:
            st.error("🚨 Fraud / Spam Message")
        else:
            st.success("✅ Legitimate Message")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.progress(int(confidence))

        st.subheader("Message")

        st.info(message)
