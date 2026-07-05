import streamlit as st
import joblib
import pickle

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
def load_files():
    try:
        model = joblib.load("fraud_model.pkl")
    except:
        with open("fraud_model.pkl", "rb") as f:
            model = pickle.load(f)

    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

    return model, vectorizer

try:
    model, vectorizer = load_files()
except Exception as e:
    st.error("Failed to load model or vectorizer.")
    st.exception(e)
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("🛡️ Myth Buster")
st.subheader("Fake / Fraud Message Detection")

st.write(
    "Paste an SMS or message below to check whether it is "
    "**Fraud/Spam** or **Legitimate**."
)

message = st.text_area(
    "Enter Message",
    height=180,
    placeholder="Congratulations! You have won ₹50,000..."
)

if st.button("Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
        st.stop()

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    st.divider()

    st.subheader("Prediction")

    # Change this if your labels are reversed
    if prediction == 1:
        st.error("🚨 Fraud / Spam Message")
    else:
        st.success("✅ Legitimate Message")

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vector).max() * 100
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))

    st.subheader("Message")
    st.info(message)
