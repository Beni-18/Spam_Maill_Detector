import streamlit as st
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")
st.write(
    "This app uses a **Naive Bayes model** trained on email text to classify "
    "messages as **Spam** or **Not Spam**."
)

# ---------------- Load Model & Vectorizer ----------------
@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# ---------------- User Input ----------------
st.subheader("✍️ Enter Email Text")

email_text = st.text_area(
    "Paste the email content below:",
    height=150,
    placeholder="Enter email text here..."
)

# ---------------- Prediction ----------------
if st.button("🔍 Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]
        confidence = model.predict_proba(email_vector).max()

        if prediction.lower() == "spam":
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM**")

        st.info(f"📊 Model confidence: **{confidence * 100:.2f}%**")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built using Scikit-learn & Streamlit")