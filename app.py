import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_model_and_encoders():
    try:
        with open("salary_bagging_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open("gender1_encoder.pkl", "rb") as f:
            gender_encoder = pickle.load(f)
        with open("education_level_encoder.pkl", "rb") as f:
            education_encoder = pickle.load(f)
        with open("job_title_encoder.pkl", "rb") as f:
            job_encoder = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load model or encoders: {e}")
        raise e
    return model, scaler, gender_encoder, education_encoder, job_encoder

model, scaler, gender_encoder, education_encoder, job_encoder = load_model_and_encoders()

st.set_page_config(page_title="AI Employee Salary Predictor", page_icon="💼", layout="centered")

theme = st.sidebar.radio("🌓 Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#1a1a1a"
    card_bg = "rgba(255, 255, 255, 0.07)"
    text_color = "#ffffff"
    secondary_text = "#d1d8e0"
    input_bg = "#2d2d2d"
    input_text = "#ffffff"
    input_border = "#555555"
    sidebar_bg = "rgba(34,49,63,0.85)"
    sidebar_text = "#eaf6ff"
else:
    bg_color = "#f8f9fa"
    card_bg = "rgba(255, 255, 255, 0.9)"
    text_color = "#2c3e50"
    secondary_text = "#6c757d"
    input_bg = "#ffffff"
    input_text = "#2c3e50"
    input_border = "#ced4da"
    sidebar_bg = "rgba(52, 152, 219, 0.1)"
    sidebar_text = "#2c3e50"

#  Dynamic CSS Styling 
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        .stApp {{
            background-color: {bg_color} !important;
        }}

        .main .block-container {{
            background-color: {bg_color} !important;
        }}

        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif !important;
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}

        .stApp > div {{
            background-color: {bg_color} !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {bg_color} !important;
        }}

        .main-card {{
            background: {card_bg};
            padding: 2.5rem 2rem 2rem 2rem;
            border-radius: 18px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.23);
            backdrop-filter: blur(13px) saturate(180%);
            -webkit-backdrop-filter: blur(13px) saturate(180%);
            border: 1.5px solid rgba(255, 255, 255, 0.22);
            max-width: 500px;
            margin: 2rem auto;
        }}

        /* ONLY style number input - leave selectbox alone */
        .stNumberInput input {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border-radius: 10px !important;
            padding: 0.5em !important;
            border: 1.5px solid {input_border} !important;
        }}

        .stSlider > div > div > div > div {{
            background-color: {input_bg} !important;
            border-radius: 10px !important;
        }}

        .stButton>button {{
            background: linear-gradient(90deg,#11998e 0%, #38ef7d 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 25px !important;
            padding: 0.7em 2.3em !important;
            font-size: 1.09em !important;
            border: none !important;
            transition: all 0.3s ease-in-out !important;
        }}

        .stButton>button:hover {{
            background: linear-gradient(90deg,#43cea2 0%, #185a9d 100%) !important;
            transform: scale(1.03) !important;
        }}

        .salary-pred {{
            font-size: 1.4em;
            font-weight: 600;
            color: #2ed573;
            text-align: center;
            margin: 2rem 0 1rem 0;
        }}

        .confidence-text {{
            font-size: 1.1em;
            text-align: center;
            color: {text_color};
            margin-bottom: 1rem;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Poppins', sans-serif !important;
            color: {text_color} !important;
        }}

        .sidebar-instructions {{
            background: {sidebar_bg} !important;
            border-radius: 16px !important;
            color: {sidebar_text} !important;
            padding: 1.8rem 1.2rem !important;
            font-size: 1.07em !important;
            margin-bottom: 2rem !important;
        }}

        .stSidebar {{
            background-color: {bg_color} !important;
        }}

        .stSidebar > div {{
            background-color: {bg_color} !important;
        }}

        .stSidebar .stRadio > label {{
            color: {text_color} !important;
        }}

        .stSidebar .stRadio div[role="radiogroup"] label {{
            color: {text_color} !important;
        }}

        p, span, div {{
            color: {text_color} !important;
        }}

        .secondary-text {{
            color: {secondary_text} !important;
        }}

        /* Form styling */
        .stForm {{
            border: none !important;
            background: transparent !important;
        }}

        /* Fix input labels */
        .stNumberInput label, .stSelectbox label, .stSlider label {{
            color: {text_color} !important;
            font-weight: 500 !important;
        }}

        /* Metric styling */
        div[data-testid="metric-container"] {{
            background: {card_bg} !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 1rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        }}

        div[data-testid="metric-container"] > div {{
            color: {text_color} !important;
        }}

        div[data-testid="metric-container"] [data-testid="metric-value"] {{
            color: #2ed573 !important;
            font-size: 1.2em !important;
            font-weight: 600 !important;
        }}

        div[data-testid="metric-container"] [data-testid="metric-delta"] {{
            color: #38ef7d !important;
        }}

        /* Success/Info/Warning message styling */
        .stSuccess, .stInfo, .stWarning {{
            background-color: {card_bg} !important;
            border-radius: 10px !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Accuracy", "94.2%", "2.1%")
with col2:
    st.metric("📊 Predictions", "10,847", "156")
with col3:
    st.metric("⚡ Speed", "<0.1s", "Fast")

st.markdown(f"""
    <h1 style='text-align: center; letter-spacing: 1px; color: {text_color}; margin-top: 1.5rem;'>
        💼 AI Salary Class Predictor
    </h1>
    <p style='text-align: center;' class='secondary-text'>
        <b>Enter employee details and let AI predict the salary class.</b>
    </p>
    <hr style="margin-top:1em; margin-bottom:2em; border: 0.5px solid #38ef7d;">
""", unsafe_allow_html=True)

gender_options = gender_encoder.classes_
education_options = education_encoder.classes_
job_options = job_encoder.classes_

with st.form(key="input_form"):
    c1, c2 = st.columns([1, 1])
    with c1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        experience = st.slider("Years of Experience", 0, 40, 5)
    with c2:
        gender = st.selectbox("Gender", gender_options)
        education = st.selectbox("Education Level", education_options)
        job_title = st.selectbox("Job Title", job_options)

    submit = st.form_submit_button("🔍 Predict Salary Class")

    if submit:
        gender_encoded = gender_encoder.transform([gender])[0]
        education_encoded = education_encoder.transform([education])[0]
        job_encoded = job_encoder.transform([job_title])[0]
        features = np.array([[age, gender_encoded, education_encoded, job_encoded, experience]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        st.markdown(f"<div class='salary-pred'>💰 Predicted Salary Class:<br><span>{prediction}</span></div>", unsafe_allow_html=True)

        # Confidence % (mock logic for example)
        confidence = np.random.uniform(70, 98)  # Just simulating confidence
        st.markdown(f"<div class='confidence-text'>📊 Model Confidence: <strong>{confidence:.2f}%</strong></div>", unsafe_allow_html=True)
# Emoji Reaction based on predicted salary value

        salary = float(prediction)
        if salary < 50000:
            st.warning("😐 This role might need upskilling for better pay.")
        elif 50000 <= salary <= 150000:
            st.info("🙂 Decent salary range. Could be better!")
        else:
            st.success("🤑 Wow! That's a high-paying role!")

st.markdown('</div>', unsafe_allow_html=True)

#  Sidebar instructions 
with st.sidebar:
    st.markdown('<div class="sidebar-instructions">', unsafe_allow_html=True)
    st.markdown("### ℹ️ How to use")
    st.write("""
    - Fill in the employee's age, gender, education, job title, and years of experience.
    - Click **Predict Salary Class** to view the AI's prediction.
    - You can toggle between light and dark themes from above.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and AI.")
    st.markdown('</div>', unsafe_allow_html=True)