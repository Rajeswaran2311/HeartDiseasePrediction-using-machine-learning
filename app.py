import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()
# Define the home function
def home():
    st.write("## Introduction")
    imageha = mpimg.imread('heart.jpg')     
    st.image(imageha)
    st.write("This app uses Random Forest Classifier to predict whether a person has heart disease or not based on some clinical and demographic features.")
   
    data=pd.read_csv('heart.csv')
    st.markdown('**Glimpse of dataset**')
    st.write(data.head(5))
    st.write("'cp' - chest pain")
    st.write("'testbps' - resting blood pressure (in mm Hg on admission in hospital)")
    st.write("'chol' - serum cholestrol in mg/dl")
    st.write("'fbs' - (fasting blood sugar > 120mg/dl) 1 = true, 0 = false")
    st.write("'restecg'- Rest ECG")
    st.write("'exang' - exercise induced angina")
    st.write("'oldpeak' - ST depression induced by exercise related to rest")
    st.write("'slope' - the slope of the peak exercise ST segment")
    st.write("'ca' - number of major vessels(0-3) colored by flourosopy")
    st.write("'thal' - (0-3) 3 = normal; 6 = fixed defect; 7 = reversable defect")
    st.write("'target' - 1 or 0")
    st.info("Please select a tab on the left to get started.")

# Define the prediction function
def prediction():
    
    st.write("Please fill in the following information to get a prediction:")
    
    # Define the input fields
    age = st.number_input("Age", value=50, min_value=1, max_value=100)
    sex = st.radio("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120, min_value=0, max_value=300)
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=200, min_value=0, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
    restecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", value=150, min_value=0, max_value=300)
    exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Flouroscopy", options=["0", "1", "2", "3"])
    thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

    #following lines create boxes in which user can enter data required to make prediction
    
 
    
    
    # Map the input values to numeric values
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 0}
    fbs_map = {"Yes": 1, "No": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
# Create a DataFrame with the input values
    data = pd.DataFrame({
    "age": [age],
    "sex": [sex_map[sex]],
    "cp": [cp_map[cp]],
    "trestbps": [trestbps],
    "restecg": [restecg_map[restecg]],
    "chol": [chol],
    "fbs": [fbs_map[fbs]],

    "thalach": [thalach],
    "exang": [exang_map[exang]],
    "oldpeak": [oldpeak],
    "slope": [slope_map[slope]],
    "ca": [int(ca)],
    "thal": [thal_map[thal]]
    })

# Load the saved logistic regression model
    model = pickle.load(open("Random_forest_model.pkl", "rb"))

# Get the model prediction
    prediction = model.predict(data)
    

# Show the prediction result
    st.write("### Prediction Result")
    if st.button("Predict"): 
        if prediction[0] == 0:
            st.success("**You have no Symptoms of getting a heart disease!**")
        else:
            st.warning("**Warning! You have high risk of getting a heart attack!**")
    
def visualization():
    st.write("## Exploratory Data Visualization")
    st.write("The following charts show some visualizations of the heart disease dataset.")

# Load the heart disease dataset
    df = pd.read_csv("heart.csv")

# Show the count of heart disease cases and non-cases
    st.write("### Heart Disease Cases")
    cases = df["target"].value_counts()
    st.write(f"There are {cases[1]} cases of heart disease and {cases[0]} cases of no heart disease in the dataset.")
    st.bar_chart(cases)

# Show the distribution of age
    st.write("### Age Distribution")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.histplot(data=df, x="age", hue="target", multiple="stack", bins=20)
    st.pyplot()

# Show the correlation between features
    st.write("### Feature Correlation")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    st.pyplot()


    



def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:")
    st.markdown("<h1 style='text-align: center; color: white;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)
# Create the tab layout
    tabs = ["Home", "Prediction", "Exploratory Data Visualization"]
    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    elif page == "Exploratory Data Visualization":
        visualization()
   
main()

