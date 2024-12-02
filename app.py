import pickle
import pandas as pd
import streamlit as st

# Correct the model path
model_path = "rf_model.pkl"  # Ensure this matches your model file

# Try loading the model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.write(f"Model loaded successfully. Type: {type(model)}")
    
    # Check if the model has the 'predict' method after loading
    if hasattr(model, 'predict'):
        st.write("The model has a 'predict' method.")
    else:
        st.error("Loaded object is not a valid model. Check 'rf_model.pkl'.")
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found. Please ensure it's in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Function to get user input for the dataset
def user_input_features():
    Age = st.sidebar.number_input("Age", min_value=18, max_value=65, step=1)
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    EducationBackground = st.sidebar.selectbox("Education Background", ["Science", "Commerce", "Arts", "Others"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    EmpDepartment = st.sidebar.selectbox("Department", ["HR", "Finance", "R&D", "Sales", "IT"])
    EmpJobRole = st.sidebar.selectbox("Job Role", ["Manager", "Executive", "Analyst", "Technician", "Clerk"])
    BusinessTravelFrequency = st.sidebar.selectbox("Business Travel Frequency", ["Rarely", "Frequently", "Never"])
    DistanceFromHome = st.sidebar.number_input("Distance From Home (km)", min_value=0, max_value=100, step=1)
    EmpEducationLevel = st.sidebar.slider("Education Level (1-5)", min_value=1, max_value=5, step=1)
    EmpEnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    EmpHourlyRate = st.sidebar.number_input("Hourly Rate", min_value=10, max_value=100, step=1)
    EmpJobInvolvement = st.sidebar.slider("Job Involvement (1-5)", min_value=1, max_value=5, step=1)
    EmpJobLevel = st.sidebar.slider("Job Level (1-5)", min_value=1, max_value=5, step=1)
    EmpJobSatisfaction = st.sidebar.slider("Job Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    NumCompaniesWorked = st.sidebar.number_input("Number of Companies Worked", min_value=0, max_value=10, step=1)
    OverTime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    EmpLastSalaryHikePercent = st.sidebar.number_input("Last Salary Hike Percent", min_value=0, max_value=100, step=1)
    EmpRelationshipSatisfaction = st.sidebar.slider("Relationship Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    TotalWorkExperienceInYears = st.sidebar.number_input("Total Work Experience (Years)", min_value=0, max_value=50, step=1)
    TrainingTimesLastYear = st.sidebar.number_input("Training Times Last Year", min_value=0, max_value=10, step=1)
    EmpWorkLifeBalance = st.sidebar.slider("Work-Life Balance (1-5)", min_value=1, max_value=5, step=1)
    ExperienceYearsAtThisCompany = st.sidebar.number_input("Experience Years At Company", min_value=0, max_value=50, step=1)
    ExperienceYearsInCurrentRole = st.sidebar.number_input("Experience Years In Current Role", min_value=0, max_value=50, step=1)
    YearsSinceLastPromotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=50, step=1)
    YearsWithCurrManager = st.sidebar.number_input("Years With Current Manager", min_value=0, max_value=50, step=1)
    Attrition = st.sidebar.selectbox("Attrition", ["Yes", "No"])

    # Combine inputs into a dataframe
    data = {
        'Age': Age,
        'Gender': Gender,
        'EducationBackground': EducationBackground,
        'MaritalStatus': MaritalStatus,
        'EmpDepartment': EmpDepartment,
        'EmpJobRole': EmpJobRole,
        'BusinessTravelFrequency': BusinessTravelFrequency,
        'DistanceFromHome': DistanceFromHome,
        'EmpEducationLevel': EmpEducationLevel,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpHourlyRate': EmpHourlyRate,
        'EmpJobInvolvement': EmpJobInvolvement,
        'EmpJobLevel': EmpJobLevel,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'EmpRelationshipSatisfaction': EmpRelationshipSatisfaction,
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'Attrition': Attrition
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Ensure the model is loaded before proceeding
if 'model' in locals():
    # Load user input
    inx = user_input_features()

    # Display input
    st.subheader("User Input:")
    st.write(inx)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(inx)
        st.subheader("Prediction:")
        st.write(prediction[0])

else:
    st.error("Model is not loaded. Please check the file path or loading process.")


