import streamlit as st
import pickle
import pandas as pd

# Load the trained pipeline model (with preprocessing inside)
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

st.title("💼 Salary Predictor App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Input Fields
Age = st.sidebar.slider("Age", 18, 65, 30)
Education_Level = st.sidebar.selectbox("Education Level", [
    "High School", "Bachelors", "Masters", "PhD"
])
Gender = st.sidebar.selectbox('Gender', ['Male', 'Female','Others'])
Years_of_Experience = st.sidebar.slider("Years of Experience", 0, 50, 1)
Job_Title = st.sidebar.selectbox('Job Title', [
    'Software Engineer', 'Data Analyst', 'Others', 'Sales Associate',
    'Marketing Analyst', 'Product Manager', 'Sales Manager',
    'Marketing Coordinator', 'Software Developer', 'Financial Analyst',
    'Operations Manager', 'Marketing Manager', 'Sales Director',
    'Financial Manager', 'Product Designer', 'Data Scientist',
    'Sales Executive', 'Director of Marketing', 'Senior Data Scientist',
    'Digital Marketing Manager', 'Web Developer', 'Research Director',
    'Senior Software Engineer', 'Content Marketing Manager',
    'Sales Representative', 'Research Scientist', 'Junior Software Developer',
    'Junior Web Developer', 'Junior HR Generalist', 'Senior HR Generalist',
    'Senior Research Scientist', 'Junior Sales Representative',
    'Junior Marketing Manager', 'Senior Product Marketing Manager',
    'Junior Software Engineer', 'Senior Human Resources Manager',
    'Junior HR Coordinator', 'Director of HR', 'Software Engineer Manager',
    'Back end Developer', 'Senior Project Engineer', 'Full Stack Engineer',
    'Front end Developer', 'Front End Developer', 'Director of Data Science',
    'Human Resources Coordinator', 'Junior Sales Associate',
    'Human Resources Manager', 'Receptionist', 'Marketing Director'
])

# Prepare input data
input_df = pd.DataFrame({
    'Age': [Age],
    'Gender': [Gender],
    'Education Level': [Education_Level],
    'Job Title': [Job_Title],
    'Years of Experience': [Years_of_Experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
