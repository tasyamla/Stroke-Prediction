import streamlit as st
import pandas as pd
import pickle

# Define the Streamlit app
def app():
    st.title('Stroke Prediction')

    st.subheader('Input Data')
    input_data = user_input()

    st.subheader('User Input')
    st.write(input_data)

    # Load the pipeline
    with open("model.pkl", "rb") as file_1:
        model = pickle.load(file_1)

    # Prediksi Stroke
    if st.button('Predict Stroke Now'):
        stroke_prediction = model.predict(input_data)
    
        st.subheader('Prediction Result:')
        if stroke_prediction[0] == 1:
            st.write('Stroke')
        else:
            st.write('No Stroke')

    else:
        st.write('Prediction Result:')
    

# Function to get user input
def user_input():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
    Residence_type = st.selectbox('Residence Type', ["Rural", "Urban"])
    avg_glucose_level = st.number_input('Average Glucose Level', value=100.0)
    bmi = st.number_input('BMI', value=20.0)
    smoking_status = st.selectbox('Smoking Status', ["formerly smoked", "never smoked", "smokes", "Unknown"])

    data = {
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [Residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    }
    
    return pd.DataFrame(data)

# Run the Streamlit app
if __name__ == '__main__':
    app()
