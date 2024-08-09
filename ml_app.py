import streamlit as st 
import joblib
import os
import numpy as np

attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - sudden weight loss 1.Yes, 2.No.
    - weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital thrush 1.Yes, 2.No.
    - visual blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - delayed healing 1.Yes, 2.No.
    - partial paresis 1.Yes, 2.No.
    - muscle stiffness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.
"""

label_dict = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}
target_label_map = {"Negative": 0, "Positive": 1}

def get_fvalue(val):
    return label_dict.get(val, None)

def get_value(val, my_dict):
    return my_dict.get(val, None)

# Load ML Models
@st.cache_data
def load_model(model_file):
    if os.path.exists(model_file):
        return joblib.load(open(model_file, "rb"))
    else:
        st.error(f"Model file {model_file} not found!")
        return None

def run_ml_app():
    st.subheader("Machine Learning Section")
    loaded_model = load_model("models/logistic_regression_model_diabetes_21_oct_2020.pkl")

    if loaded_model is None:
        st.error("Model not loaded.")
        return

    with st.expander("Attributes Info"):
        st.markdown(attrib_info, unsafe_allow_html=True)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100)
        gender = st.radio("Gender", ("Female", "Male"))
        polyuria = st.radio("Polyuria", ["No", "Yes"])
        polydipsia = st.radio("Polydipsia", ["No", "Yes"]) 
        sudden_weight_loss = st.selectbox("Sudden_weight_loss", ["No", "Yes"])
        weakness = st.radio("Weakness", ["No", "Yes"]) 
        polyphagia = st.radio("Polyphagia", ["No", "Yes"]) 
        genital_thrush = st.selectbox("Genital_thrush", ["No", "Yes"])

    with col2:
        visual_blurring = st.selectbox("Visual_blurring", ["No", "Yes"])
        itching = st.radio("Itching", ["No", "Yes"]) 
        irritability = st.radio("Irritability", ["No", "Yes"]) 
        delayed_healing = st.radio("Delayed_healing", ["No", "Yes"]) 
        partial_paresis = st.selectbox("Partial_paresis", ["No", "Yes"])
        muscle_stiffness = st.radio("Muscle_stiffness", ["No", "Yes"]) 
        alopecia = st.radio("Alopecia", ["No", "Yes"]) 
        obesity = st.select_slider("Obesity", ["No", "Yes"]) 

    with st.expander("Your Selected Options"):
        result = {
            'age': age,
            'gender': gender,
            'polyuria': polyuria,
            'polydipsia': polydipsia,
            'sudden_weight_loss': sudden_weight_loss,
            'weakness': weakness,
            'polyphagia': polyphagia,
            'genital_thrush': genital_thrush,
            'visual_blurring': visual_blurring,
            'itching': itching,
            'irritability': irritability,
            'delayed_healing': delayed_healing,
            'partial_paresis': partial_paresis,
            'muscle_stiffness': muscle_stiffness,
            'alopecia': alopecia,
            'obesity': obesity
        }
        st.write(result)
        
        encoded_result = []
        for key, value in result.items():
            if key == 'gender':
                encoded_result.append(get_value(value, gender_map))
            else:
                encoded_result.append(get_fvalue(value))
        
        st.write("Encoded features:", encoded_result)
        
    with st.expander("Prediction Results"):
        single_sample = np.array(encoded_result).reshape(1, -1)
        st.write("Single sample shape:", single_sample.shape)
        st.write("Single sample content:", single_sample)

        try:
            prediction = loaded_model.predict(single_sample)
            pred_prob = loaded_model.predict_proba(single_sample)
            st.write("Prediction:", prediction)
            if prediction[0] == 1:
                st.warning("Positive Risk - {}".format(prediction[0]))
                pred_probability_score = {"Negative DM": pred_prob[0][0] * 100, "Positive DM": pred_prob[0][1] * 100}
                st.subheader("Prediction Probability Score")
                st.json(pred_probability_score)
            else:
                st.success("Negative Risk - {}".format(prediction[0]))
                pred_probability_score = {"Negative DM": pred_prob[0][0] * 100, "Positive DM": pred_prob[0][1] * 100}
                st.subheader("Prediction Probability Score")
                st.json(pred_probability_score)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    run_ml_app()
