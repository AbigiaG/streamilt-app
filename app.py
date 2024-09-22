import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Define expected features (replace with your actual features)
feature_names = [
    'location_type', 
    'cellphone_access', 
    'household_size', 
    'age_of_respondent', 
    'gender_of_respondent', 
    'relationship_with_head', 
    'marital_status', 
    'education_level', 
    'job_type',
    'salary' 
]

# Function to get user input
def get_user_input():
  age_of_respondent = st.number_input('Age', min_value=0, max_value=100, value=25)
  salary = st.number_input('Salary', min_value=0, value=50000)
  gender_of_respondent = st.selectbox('Gender', options=['Male', 'Female'])
  education_level = st.selectbox('Education Level', options=['High School', 'Bachelor', 'Master', 'PhD'])
  # Return user inputs as a dictionary
  return {
       
        'age_of_respondent': age_of_respondent,
        'gender_of_respondent': gender_of_respondent,
       'education_level': education_level,
       
    }

# Function to preprocess user input 
def preprocess_input(user_input):
    # Assuming 'age' and 'salary' are collected in the actual application
    age = user_input['age_of_respondent']  
    # Placeholder for salary, replace with actual input collection
    salary = st.number_input("Salary", min_value=0, value=50000)  
    gender = user_input['gender_of_respondent']
    education_level = user_input['education_level']

    # Encode categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    education_encoded = education_mapping[education_level]
# Create DataFrame
    input_data = {
    'age': [age],
    'salary': [salary],
    'gender': [gender_encoded],
    'education_level': [education_encoded]
    }
    return input_data
    # Return preprocessed input

    
  

# App layout
st.title('Employee Attrition Prediction')

st.write("""
### Please enter the following details to predict employee attrition:
""")

user_input = get_user_input()
input_df = preprocess_input(user_input)

if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.write(f'**Prediction:** {"Attrition" if prediction[0] == 1 else "No Attrition"}')
    st.write(f'**Prediction Probability:** {prediction_proba}')': ['Married', 'Single', 'Widowed', 'Single'],
        'education_level': ['Primary', 'Secondary', 'No formal education', 'Vocational'],
        'job_type': ['Self employed', 'Government Dependent', 'Informally employed', 'Self employed'],
        'bank_account': [1, 0, 0, 1]  # Target
    }


