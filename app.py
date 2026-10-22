
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="MindEase AI | Exam Wellness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f3f4ff, #f0fbf8);
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #4338ca, #7c3aed, #0f9f91);
    padding: 35px 30px;
    border-radius: 24px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 10px;
}
.hero p {
    color: #f0edff;
    font-size: 17px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    border-color: #e5e7eb;
}
div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #0f9f91);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 700;
    min-height: 48px;
}
div.stButton > button:hover {
    color: white;
    border: 1px solid #4f46e5;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
}
.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🧠 MindEase AI</h1>
    <p>Your personal exam wellness companion</p>
    <p>Understand your study habits, reflect on exam stress,
    and discover simple ways to feel more prepared.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "🌿 **Small steps. Better habits. A calmer exam experience.**"
)

st.info(
    "Educational prototype only. This tool does not diagnose "
    "anxiety or replace professional advice."
)

# ---------- INPUT FORM ----------
st.subheader("✨ Tell us about your routine")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "📚 Daily study hours",
            min_value=0, max_value=12, value=4
        )
        heart_rate = st.number_input(
            "💓 Heart rate (if known)",
            min_value=40, max_value=200, value=80,
            help="Use a measured value if available."
        )

    with col2:
        sleep_hours = st.slider(
            "😴 Daily sleep hours",
            min_value=0, max_value=12, value=7
        )
        exam_stress = st.slider(
            "😟 Self-rated exam stress",
            min_value=1, max_value=10, value=5,
            help="1 = very low, 10 = very high"
        )

    st.caption(
        "Your answers are used for this prediction. "
        "This demo does not save them to a database."
    )

    check = st.button(
        "✨ Check My Stress Indicator",
        use_container_width=True
    )

# ---------- PREDICTION ----------
if check:
    model_path = Path(__file__).with_name(
        "exam_anxiety_model.pkl"
    )

    if not model_path.exists():
        st.error(
            "Model file not found. Keep "
            "exam_anxiety_model.pkl beside app.py."
        )
    else:
        try:
            with open(model_path, "rb") as file:
                model = pickle.load(file)

            input_data = pd.DataFrame(
                [[study_hours, sleep_hours,
                  heart_rate, exam_stress]],
                columns=[
                    "study_hours", "sleep_hours",
                    "heart_rate", "exam_stress"
                ]
            )

            prediction = model.predict(input_data)[0]

            st.divider()
            st.subheader("📊 Your Result")

            if prediction == 1:
                st.warning("🌼 Higher Anxiety Indicator")
                st.write(
                    "Your model returned the higher-indicator "
                    "class. Try a short break, slow comfortable "
                    "breathing, and breaking revision into "
                    "smaller tasks."
                )
                st.progress(75, text="Higher-indicator category")
            else:
                st.success("🌱 Lower Anxiety Indicator")
                st.write(
                    "Your model returned the lower-indicator "
                    "class. Continue with balanced revision, "
                    "regular breaks, and sufficient sleep."
                )
                st.progress(30, text="Lower-indicator category")

            st.caption(
                "The progress bar represents a category "
                "illustration, NOT a measured anxiety score "
                "or prediction probability."
            )

            # Routine insights
            st.subheader("💡 Your Wellness Tips")

            tip1, tip2, tip3 = st.columns(3)

            with tip1:
                st.markdown("#### 📚 Study")
                if study_hours < 2:
                    st.write(
                        "Try a manageable revision plan "
                        "with short focused sessions."
                    )
                elif study_hours > 8:
                    st.write(
                        "Schedule breaks between long "
                        "study sessions."
                    )
                else:
                    st.write(
                        "Keep a consistent study routine "
                        "and take regular breaks."
                    )

            with tip2:
                st.markdown("#### 😴 Sleep")
                if sleep_hours < 7:
                    st.write(
                        "Consider allowing more time "
                        "for sleep and recovery."
                    )
                else:
                    st.write(
                        "Maintain a regular sleep schedule."
                    )

            with tip3:
                st.markdown("#### 🧘 Reset")
                st.write(
                    "Relax your shoulders, breathe gently, "
                    "and take a short screen-free break."
                )

            st.info(
                "This is an experimental ML prediction, "
                "not a medical diagnosis. The result depends "
                "on the model and its training data."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------- FOOTER ----------
st.divider()
st.markdown(
    "<p style='text-align:center; color:#64748b;'>"
    "Built with Python • Streamlit • Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
