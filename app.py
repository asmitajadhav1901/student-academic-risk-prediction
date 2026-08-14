import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title & Description
st.title("🎓 Student Performance Predictor")
st.write(
    "Predict student marks/scores based on daily study time and number of courses "
    "using the trained **K-Nearest Neighbors Regressor** model."
)

st.divider()

# Cache the model loading for better performance
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ `model.pkl` not found! Please place the `model.pkl` file in the same directory as `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Input Section
st.header("📋 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    number_courses = st.number_input(
        "Number of Courses",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Select the total number of enrolled courses."
    )

with col2:
    time_study = st.number_input(
        "Study Time (Hours/Day)",
        min_value=0.0,
        max_value=24.0,
        value=4.5,
        step=0.25,
        help="Enter average daily study hours."
    )

st.divider()

# Prediction Action
if st.button("🚀 Predict Marks", use_container_width=True, type="primary"):
    # Construct DataFrame with exact feature names matching model training
    input_df = pd.DataFrame(
        [[number_courses, time_study]], 
        columns=['number_courses', 'time_study']
    )
    
    # Run Prediction
    try:
        prediction = model.predict(input_df)[0]
        
        # Display Result
        st.subheader("🎯 Prediction Result")
        st.metric(
            label="Estimated Score", 
            value=f"{prediction:.2f}"
        )
        
        st.success(
            f"A student studying **{time_study} hrs/day** across **{number_courses} courses** "
            f"is predicted to score approximately **{prediction:.2f}**."
        )
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Sidebar Information
st.sidebar.header("ℹ️ Model Info")
st.sidebar.info(
    """
    **Model Type:** `KNeighborsRegressor`  
    **Features:**  
    - `number_courses`  
    - `time_study`  
    """
)
