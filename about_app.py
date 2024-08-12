import streamlit as st

def about():
    st.subheader('About the Attributes')
    
    with st.expander('Information Box:'):
        attribute_list = ['polyuria', 'polydipsia', 'sudden_weight_loss',
                          'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
                          'itching', 'irritability', 'delayed_healing', 'partial_paresis',
                          'muscle_stiffness', 'alopecia', 'obesity']
        
        selected_attribute = st.selectbox("Select attribute", attribute_list)
        
        attribute_info = {
            'polyuria': "Polyuria is a condition characterized by excessive urination. "
                        "It is often a symptom of diabetes and can be caused by high blood sugar levels.",
            
            'polydipsia': "Polydipsia is excessive thirst and one of the early symptoms of diabetes. "
                          "It can be a sign of high blood sugar levels and should not be ignored.",
            
            'sudden_weight_loss': "Sudden weight loss can be a symptom of diabetes. It occurs because the body "
                                  "is unable to get energy from glucose and starts burning muscle and fat instead.",
            
            'weakness': "Weakness can be a common symptom of diabetes, often due to fluctuations in blood sugar levels.",
            
            'polyphagia': "Polyphagia refers to excessive hunger or increased appetite, which is a common symptom of diabetes.",
            
            'genital_thrush': "Genital thrush is a yeast infection that can occur in both men and women, "
                              "and is more common in people with diabetes.",
            
            'visual_blurring': "Visual blurring is a common symptom of diabetes, often caused by high blood sugar levels "
                               "leading to swelling in the eye lens.",
            
            'itching': "Itching can be a symptom of diabetes, often caused by dry skin or poor blood circulation.",
            
            'irritability': "Irritability can be a psychological symptom associated with diabetes due to fluctuating blood sugar levels.",
            
            'delayed_healing': "Delayed healing of wounds is common in people with diabetes, often due to poor blood circulation "
                               "and high blood sugar levels.",
            
            'partial_paresis': "Partial paresis is a condition of muscle weakness or partial paralysis that can occur in individuals with diabetes.",
            
            'muscle_stiffness': "Muscle stiffness can be a symptom associated with diabetes, often related to nerve damage (neuropathy).",
            
            'alopecia': "Alopecia, or hair loss, can be a symptom of diabetes due to poor blood circulation or hormonal changes.",
            
            'obesity': "Obesity is a significant risk factor for developing diabetes and can complicate its management."
        }
        
        collective_source_url = "https://www.diabetes.co.uk"
        
        if selected_attribute:
            st.write(attribute_info[selected_attribute])
            st.markdown(f"For more information, visit [Diabetes.co.uk]({collective_source_url}) and search for '{selected_attribute}'.")

# Example usage
if __name__ == "__main__":
    about()
