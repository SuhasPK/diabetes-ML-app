import streamlit as st  # type: ignore
import joblib  # type: ignore
import os
import numpy as np
import pickle

# Attribute Information
attribute_info = """
## Attribute Information

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

# Maps for encoding features
label_dict = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}
target_label_map = {"Negative": 0, "Positive": 1}

def get_fvalue(val):
    return label_dict.get(val, None)

def get_value(val, my_dict):
    return my_dict.get(val, None)

@st.cache_resource
def load_model(model_file):
    if os.path.exists(model_file):
        model = joblib.load(open(model_file, "rb"))
        st.write("Model loaded successfully.")
        return model
    else:
        st.error(f"Model file {model_file} not found!")
        return None

def run_ml_app():
    st.subheader("Machine Learning Prediction")

    with st.expander('About ML model'):
        st.markdown("""
        - The ML model used in this project for the prediction is **Logistic Regression**. 
        - Logistic regression is used for binary classification where we use the sigmoid function, which takes input as independent variables and produces a probability value between 0 and 1. 
        - Logistic regression predicts the output of a categorical dependent variable. Therefore, the outcome must be a categorical or discrete value.
        - It can be either Yes or No, 0 or 1, true or false, etc., but instead of giving the exact value as 0 and 1, it gives probabilistic values that lie between 0 and 1. 
        - There should be little to no collinearity between independent variables.
        """)

    with st.expander('Attribute Information'):
        st.markdown(attribute_info)

    # Collect user inputs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input("Age", 10, 100)
        gender = st.radio("Gender", ['Female', 'Male'])
        polyuria = st.radio("Polyuria", ['No', 'Yes'])
        polydispia = st.radio("Polydispia", ['No', 'Yes'])

    with col2:
        sudden_weight_loss = st.radio("Sudden Weight Loss", ['No', 'Yes'])
        weakness = st.radio("Weakness", ['No', 'Yes'])
        polyphagia = st.radio("Polyphagia", ['No', 'Yes'])
        genital_thrush = st.radio("Genital Thrush", ['No', 'Yes'])

    with col3:
        visual_blurring = st.radio("Visual Blurring", ['No', 'Yes'])
        itching = st.radio("Itching", ['No', 'Yes'])
        irritability = st.radio("Irritability", ['No', 'Yes'])
        delayed_healing = st.radio("Delayed Healing", ['No', 'Yes'])

    with col4:
        partial_paresis = st.radio("Partial Paresis", ['No', 'Yes'])
        muscle_stiffness = st.radio("Muscle Stiffness", ['No', 'Yes'])
        alopecia = st.radio("Alopecia", ['No', 'Yes'])
        obesity = st.radio("Obesity", ['No', 'Yes'])

    # Display user-selected options
    with st.expander("Your selected options are (json format): "):
        result = {
            'age': age,
            'gender': gender,
            'polyuria': polyuria,
            'polydispia': polydispia,
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

        # Encode the features
        encoded_result = []
        for key, value in result.items():
            if key == 'gender':
                encoded_result.append(get_value(value, gender_map))
            else:
                encoded_result.append(get_fvalue(value))

        # Display encoded features for debugging
        st.write("Encoded features:", encoded_result)

    # Make prediction
    with st.expander("Prediction Result "):
        single_sample = np.array(encoded_result).reshape(1, -1)
        st.write("Single sample shape:", single_sample.shape)
        st.write("Single sample type:", type(single_sample))

        loaded_model = load_model("models/logistic_regression.pkl")
        
        if loaded_model is None:
            st.error("Model not loaded.")
            return

        if not hasattr(loaded_model, 'predict'):
            st.error("Loaded object is not a model with a 'predict' method.")
            return

        try:
            prediction = loaded_model.predict(single_sample)
            predict_probability = loaded_model.predict_proba(single_sample)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        if prediction[0] == 1:
            st.warning("Positive Risk! {}".format(prediction[0]), icon="⚠️")
            predict_probability_score = {
                "Negative DM risk": predict_probability[0][0] * 100,
                "Positive DM risk": predict_probability[0][1] * 100
            }
            st.write(predict_probability_score)
        else:
            st.success("Negative Risk! {}".format(prediction[0]), icon="✅")
            predict_probability_score = {
                "Negative DM risk": predict_probability[0][0] * 100,
                "Positive DM risk": predict_probability[0][1] * 100
            }
            st.write(predict_probability_score)

# Run the app
if __name__ == "__main__":
    run_ml_app()
