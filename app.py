import streamlit as st
from eda_app import run_eda_app
from about_app import about
from ml_app import run_ml_app

# Set page configuration for SEO and mobile compatibility
st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    page_icon=":bar_chart:",  # Use a relevant icon or emoji
    layout="wide",  # Adjust layout as needed
    initial_sidebar_state="expanded"  # Sidebar state
)

# Inject additional HTML for SEO
st.markdown("""
    <meta name="description" content="Predict the risk of diabetes based on various health attributes using a Logistic Regression model.">
    <meta name="keywords" content="diabetes, prediction, health, machine learning, logistic regression">
    <meta name="author" content="Suhas. P. K">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta property="og:title" content="Diabetes Risk Prediction App">
    <meta property="og:description" content="Predict the risk of diabetes based on various health attributes using a Logistic Regression model.">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://app-diabetes-ml-prediction.streamlit.app/">
    <meta property="og:image" content="https://your-app-url/static/diabetes-prediction.png">
    """, unsafe_allow_html=True)

def main():
    st.title("Diabetes Risk Detection")
    st.warning("### Suhas. P. K")

    menu = ['Home', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
        st.markdown("""
            ### Welcome to the Early Stage Diabetes Risk Prediction Web App
            This web application is designed to help in the early detection of diabetes risk using various features from a dataset of diabetic and non-diabetic patients.
            
            #### Project Overview
            Diabetes is a serious condition that affects millions of people worldwide. Early detection and intervention can significantly improve the management and outcomes of diabetes.
            
            This app aims to:
            - **Analyze and visualize**: Explore the dataset using various exploratory data analysis (EDA) techniques to understand the distribution of different features and their relation to diabetes.
            - **Predict risk**: Use a Logistic Regression model to predict the risk of diabetes based on user-provided input features.
            
            #### Features
            - **EDA Section**: Perform exploratory data analysis on the dataset to gain insights and visualize distributions.
            - **ML Section**: Use a machine learning model to predict diabetes risk based on user inputs.
            - **About Section**: Get detailed information about various symptoms and their relation to diabetes.
            
            #### How to Use
            - Navigate to the **'EDA'** section to explore data distributions and descriptive statistics.
            - Go to the **'ML'** section to input your data and get predictions on diabetes risk.
            - Check the **'About'** section for detailed information about the symptoms and their relevance.
            
            #### Data Source
            The dataset used in this application is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset).
            
            #### GitHub Repository
            You can find the source code and additional documentation on [GitHub](https://github.com/SuhasPK/diabetes-ML-app).
            """)

    elif choice == 'EDA':
        run_eda_app()
    elif choice == 'ML':
        st.subheader("Machine Learning")
        run_ml_app()
    else:
        about()

if __name__ == '__main__':
    main()
