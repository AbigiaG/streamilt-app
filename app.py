import streamlit as st
import pandas as pd
import numpy as np

# Load trained model (assuming a pre-trained model is available)
# For demonstration, we're training a simple model in this code
def load_model():
    # Sample DataFrame to mock training (replace this with actual training data)
    data = {
        'location_type': ['Rural', 'Urban', 'Rural', 'Urban'],
        'cellphone_access': ['Yes', 'No', 'Yes', 'No'],
        'household_size': [3, 5, 2, 4],
        'age_of_respondent': [24, 70, 34, 26],
        'gender_of_respondent': ['Female', 'Male', 'Female', 'Male'],
        'relationship_with_head': ['Spouse', 'Head of Household', 'Child', 'Head of Household'],
        'marital_status': ['Married', 'Single', 'Widowed', 'Single'],
        'education_level': ['Primary', 'Secondary', 'No formal education', 'Vocational'],
        'job_type': ['Self employed', 'Government Dependent', 'Informally employed', 'Self employed'],
        'bank_account': [1, 0, 0, 1]  # Target
    }

    df = pd.DataFrame(data)

    # Encode categorical features
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])

    X = df.drop(columns=['bank_account'])
    y = df['bank_account']

    # Train RandomForestClassifier for the sake of this demo
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)

    return clf, label_encoder, X.columns  # Returning model, label encoder, and feature names


# Load model and other components
clf, label_encoder, feature_columns = load_model()

# App title
st.title('Machine Learning Model for financial Inclusion Prediction')

st.write("""
### Predict whether an individual has a bank account based on their characteristics.
""")

# Create input fields for the features
location_type = st.selectbox("Location Type", ['Rural', 'Urban'])
cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])
household_size = st.slider("Household Size", min_value=1, max_value=20, value=3)
age_of_respondent = st.slider("Age of Respondent", min_value=16, max_value=100, value=30)
gender_of_respondent = st.selectbox("Gender", ['Female', 'Male'])
relationship_with_head = st.selectbox("Relationship with Head", ['Head of Household', 'Spouse', 'Child', 'Other relative'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Widowed'])
education_level = st.selectbox("Education Level", ['No formal education', 'Primary', 'Secondary', 'Vocational/Specialized'])
job_type = st.selectbox("Job Type", ['Self employed', 'Government Dependent', 'Informally employed', 'Formally employed Private'])

# Create a button for prediction
if st.button("Predict"):
    # Prepare the input data in the right format
    input_data = pd.DataFrame({
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    })

    # Encode categorical features using the trained LabelEncoder
    for column in input_data.select_dtypes(include=['object']).columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])

    # Predict using the trained model
    prediction = clf.predict(input_data)[0]

    # Display the result
    if prediction == 1:
        st.success("The individual is predicted to have a bank account.")
    else:
        st.error("The individual is predicted to not have a bank account.")


