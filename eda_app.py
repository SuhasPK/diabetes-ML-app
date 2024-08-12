import streamlit as st  # type: ignore

# EDA
import numpy as np
import pandas as pd

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ignore warning
import warnings
warnings.filterwarnings("ignore")

import io

##################################################################################

def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader('Exploratory Data Analysis')
    st.markdown('Do explore all features and its relation with the classes.')
    st.markdown('Try to interprete the distribution graph.')
    st.markdown('Download option is also available.')
    df = load_data('notebooks/dataset/diabetes_data_upload.csv')
    df_encoded = load_data('notebooks/dataset/encoded_data.csv')

    submenu = st.sidebar.selectbox('Submenu', ['Descriptive', 'Plots'])
    if submenu == 'Descriptive':
        with st.expander("Data-set"):
            st.dataframe(df)

        with st.expander("Class distribution  "):
            st.dataframe(df['class'].value_counts())
            st.write("""This shows how many of them were reported DM positive and negative.""")

        with st.expander("Descriptive Summary "):
            st.dataframe(df_encoded.describe())
            st.write("""A simple statistical interpretation table.""")

        with st.expander("Gender distribution "):
            st.dataframe(df['Gender'].value_counts())
            st.write("""This shows how many of the test subjects were male and female.""")

    ######################  PLOTS  ###################################################
    
    elif submenu == 'Plots':
        st.subheader("Data visualizations")
        st.write("""Visualizations are a very powerful tool for understanding data in a simple way. Do go through all of it.""")

        with st.expander("Class Distribution "):
            class_count = df_encoded['class'].value_counts().reset_index()
            class_count.columns = ['class', 'count']
            class_count['class'] = class_count['class'].map({0: 'Negative', 1: 'Positive'})

            # Custom colors for the bars
            colors = {'Positive': 'red', 'Negative': 'green'}

            fig = px.bar(class_count, x='class', y='count',
                         labels={'class': 'Class', 'count': 'Count'},
                         title='Count vs Class',
                         template="plotly_dark",
                         color='class',  # Color the bars based on class
                         color_discrete_map=colors,
                         text='count'  # Add count numbers as text on the bars
                         )

            # Adjust the text position to be inside the bars
            fig.update_traces(textposition='inside')
            fig.update_layout(width=600, height=500)
            fig.update_traces(marker_line_width=2, marker_line_color='white', width=0.4)  # Adjust width as needed
            fig.write_image("images/Count_vs_Class.png")

            st.plotly_chart(fig)

            # Create an in-memory byte buffer
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)

            # Download button to download the plot as PNG
            st.download_button(
                label="Download plot",
                data=buf,
                file_name="Count_vs_Class.png",
                mime="image/png"
            )

        with st.expander("Gender Distribution "):
            class_count = df_encoded['gender'].value_counts().reset_index()
            class_count.columns = ['gender', 'count']
            class_count['gender'] = class_count['gender'].map({0: 'Female', 1: 'Male'})

            # Custom colors for the bars
            colors = {'Male': 'orange', 'Female': 'green'}

            fig = px.bar(class_count, x='gender', y='count',
                         labels={'gender': 'Gender', 'count': 'Count'},
                         title='Count vs Gender',
                         template="plotly_dark",
                         color='gender',  # Color the bars based on class
                         color_discrete_map=colors,
                         text='count'  # Add count numbers as text on the bars
                         )

            # Adjust the text position to be inside the bars
            fig.update_traces(textposition='inside')
            fig.update_layout(width=1000, height=600)
            fig.update_traces(marker_line_width=2, marker_line_color='white', width=0.4)  # Adjust width as needed
            fig.write_image("images/Count_vs_Gender.png")

            st.plotly_chart(fig)

            # Create an in-memory byte buffer
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)

            # Download button to download the plot as PNG
            st.download_button(
                label="Download plot",
                data=buf,
                file_name="Count_vs_Gender.png",
                mime="image/png"
            )

        with st.expander("Distribution plot "):
            attribute_list = ['polyuria', 'polydipsia', 'sudden_weight_loss',
                              'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
                              'itching', 'irritability', 'delayed_healing', 'partial_paresis',
                              'muscle_stiffness', 'alopecia', 'obesity']

            selected_attribute = st.selectbox("Select attribute", attribute_list)

            def distribution_plot(attribute):
                # Check if the attribute is in the DataFrame
                if attribute not in df_encoded.columns:
                    raise ValueError(f"Attribute '{attribute}' not found in the DataFrame.")
        
                # Create a copy of the DataFrame and create age groups
                temp_df = df_encoded.copy()
                bins = [0, 20, 40, 60, 80, 100]
                labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
                temp_df['age_group'] = pd.cut(temp_df['age'], bins=bins, labels=labels, right=False)

                # Convert class and gender columns
                temp_df['class'] = temp_df['class'].map({0: 'Negative', 1: 'Positive'})
                temp_df['gender'] = temp_df['gender'].map({0: 'Female', 1: 'Male'})

                # Filter and create dataframes for males and females
                male_df = temp_df[temp_df['gender'] == 'Male']
                female_df = temp_df[temp_df['gender'] == 'Female']

                # Create DataFrame-1: All males with age groups and attribute
                dataframe_1 = male_df.groupby(['age_group', 'class'])[attribute].sum().reset_index(name='count')

                # Create DataFrame-2: All females with age groups and attribute
                dataframe_2 = female_df.groupby(['age_group', 'class'])[attribute].sum().reset_index(name='count')

                # Create subplots: one row, two columns
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Male', 'Female'),
                    shared_yaxes=True,
                    horizontal_spacing=0.1
                )

                # Plot for Male
                fig.add_trace(go.Bar(
                    x=dataframe_1[dataframe_1['class'] == 'Positive']['age_group'],
                    y=dataframe_1[dataframe_1['class'] == 'Positive']['count'],
                    name='Positive',
                    marker_color='red'
                ), row=1, col=1)

                fig.add_trace(go.Bar(
                    x=dataframe_1[dataframe_1['class'] == 'Negative']['age_group'],
                    y=dataframe_1[dataframe_1['class'] == 'Negative']['count'],
                    name='Negative',
                    marker_color='green'
                ), row=1, col=1)

                # Plot for Female
                fig.add_trace(go.Bar(
                    x=dataframe_2[dataframe_2['class'] == 'Positive']['age_group'],
                    y=dataframe_2[dataframe_2['class'] == 'Positive']['count'],
                    name='Positive',
                    marker_color='orange'
                ), row=1, col=2)

                fig.add_trace(go.Bar(
                    x=dataframe_2[dataframe_2['class'] == 'Negative']['age_group'],
                    y=dataframe_2[dataframe_2['class'] == 'Negative']['count'],
                    name='Negative',
                    marker_color='blue'
                ), row=1, col=2)

                # Update layout
                fig.update_layout(
                    title_text=f'Distribution of {attribute} by Age Group and Gender',
                    xaxis_title='Age Group',
                    yaxis_title='Count',
                    barmode='group',
                    template='plotly_dark',
                    legend_title='Class',
                    xaxis=dict(
                        tickangle=-45
                    ),
                    xaxis2=dict(
                        tickangle=-45
                    )   
                )
                fig.update_layout(width=1000, height=600)
                fig.update_traces(marker_line_width=1, marker_line_color='white', width=0.3)
                filename = f"images/distribution_{attribute}_class_gender_age_group.png"
                fig.write_image(filename)
        
                # Show plot
                st.plotly_chart(fig)

                # Create an in-memory byte buffer
                buf = io.BytesIO()
                fig.write_image(buf, format="png")
                buf.seek(0)

                # Download button to download the plot as PNG
                st.download_button(
                    label="Download plot",
                    data=buf,
                    file_name=f"Distribution_of_{attribute}_by_age_and_gender.png",
                    mime="image/png"
                )

            if selected_attribute:
                distribution_plot(selected_attribute)

if __name__ == '__main__':
    run_eda_app()
