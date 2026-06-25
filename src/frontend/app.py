import streamlit as st
import requests


st.set_page_config(page_title="AI Layoff Risk Prediction")

st.title("AI Impact on Jobs & Layoff Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    education_level = st.selectbox(
        "Education Level", ["High School", "Bachelor's", "Master's", "PhD"]
    )

    years_of_experience = st.number_input(
        "Years of Experience", min_value=0, max_value=50, value=5
    )

    industry = st.selectbox(
        "Industry",
        [
            "Finance",
            "Manufacturing",
            "Retail",
            "Healthcare",
            "Logistics",
            "Education",
            "Telecom",
            "IT",
        ],
    )
    job_role = st.selectbox(
        "Job Role",
        [
            "Accountant",
            "Production Supervisor",
            "Store Manager",
            "Auditor",
            "Network Engineer",
            "Software Engineer",
            "ML Engineer",
            "Inventory Analyst",
            "Dispatcher",
            "Health Analyst",
            "Support Specialist",
            "Sales Associate",
            "Academic Coordinator",
            "Financial Analyst",
            "Operator",
            "Research Assistant",
            "Quality Engineer",
            "Teacher",
            "Warehouse Manager",
            "Supply Chain Analyst",
            "Operations Analyst",
            "Medical Assistant",
            "Nurse",
            "Data Analyst",
        ],
    )

    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    job_level = st.selectbox(
        "Job Level",
        [
            "Entry",
            "Mid",
            "Senior",
        ],
    )

with col2:
    routine_task_percentage = st.slider("Routine Task Percentage", 0, 100, 50)

    creativity_requirement = st.slider("Creativity Requirement", 0, 100, 50)

    human_interaction_level = st.slider("Human Interaction Level", 0, 100, 50)

    ai_adoption_level = st.selectbox("AI Adoption Level", ["Low", "Medium", "High"])

    number_of_ai_tools_used = st.number_input(
        "Number of AI Tools Used", min_value=0, max_value=20, value=0
    )

    ai_usage_hours_per_week = st.number_input(
        "AI Usage Hours Per Week", min_value=0, max_value=100, value=0
    )

    tasks_automated_percentage = st.slider("Tasks Automated Percentage", 0, 100, 0)

    ai_training_hours = st.number_input(
        "AI Training Hours", min_value=0, max_value=500, value=0
    )

st.divider()


if st.button("Predict Layoff Risk", use_container_width=True):
    payload = {
        "Age": age,
        "Education_Level": education_level,
        "Years_of_Experience": years_of_experience,
        "Industry": industry,
        "Job_Role": job_role,
        "Company_Size": company_size,
        "Job_Level": job_level,
        "Routine_Task_Percentage": routine_task_percentage,
        "Creativity_Requirement": creativity_requirement,
        "Human_Interaction_Level": human_interaction_level,
        "AI_Adoption_Level": ai_adoption_level,
        "Number_of_AI_Tools_Used": number_of_ai_tools_used,
        "AI_Usage_Hours_Per_Week": ai_usage_hours_per_week,
        "Tasks_Automated_Percentage": tasks_automated_percentage,
        "AI_Training_Hours": ai_training_hours,
    }

    try:
        response = requests.post("http://127.0.0.1:8003/predict-risk", json=payload)

        result = response.json()

        if result is not None:
            st.success(f"Predicted Layoff Risk: {result.get('prediction')}")
        else:
            st.error("API returned None")

    except Exception as e:
        st.error(str(e))
