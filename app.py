import streamlit as st # type: ignore
import streamlit.components.v1 as stc # type: ignore
from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Early stage DM risk prediction web app. </h1>
		<h4 style="color:white;text-align:center;"> Suhas. P. K </h4>
		</div>
		"""

def main():
    st.title("Diabetes Risk detection")
    stc.html(html_temp)

    menu = ['Home', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
        st.markdown("""
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App
			""")
        st.link_button("Data Source","https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset")
        st.link_button("My Github repo", "https://github.com/SuhasPK/diabetes-ML-app")
    elif choice == 'EDA':
        run_eda_app()
    elif choice == 'ML':
        run_ml_app()
    else:
        st.subheader('About')


if __name__ == '__main__':
    main()