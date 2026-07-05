import streamlit as st
import joblib

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Myth Buster",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("❌ Failed to load the ML model.")
    st.exception(e)
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("🛡️ Myth Buster")
st.subheader("Fake / Fraud Message Detection")

st.write(
    "Paste an SMS or message below. The trained Machine Learning model "
    "will classify it as **Legitimate** or **Fraud/Spam**."
)

st.divider()

# -----------------------------
# User Input
# -----------------------------
message = st.text_area(
    "Enter Message",
    height=200,
    placeholder="Example: Congratulations! You have won ₹50,000..."
)

# -----------------------------
# Predict
# -----------------------------
if st.button("🔍 Analyze Message"):

    if not message.strip():
        st.warning("Please enter a message.")
        st.stop()

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector)[0][prediction] * 100

    st.divider()

    st.subheader("Prediction")

    if prediction == 1:
        st.error("🚨 Fraud / Spam Message")
    else:
        st.success("✅ Legitimate Message")

    st.metric("Confidence", f"{confidence:.2f}%")

    st.progress(int(confidence))

    st.subheader("Entered Message")
    st.info(message)
