import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

def load_data(data):
    """Load dataset from a CSV file."""
    df = pd.read_csv(data)
    return df

def run_eda_app():
    """Run Exploratory Data Analysis (EDA) app."""
    st.subheader('Exploratory Data Analysis')
    st.markdown('Explore all features and their relationship with the classes.')
    st.markdown('Try to interpret the distribution graphs.')
    st.markdown('Download option is available for plots.')

    # Load datasets
    df = load_data('notebooks/dataset/diabetes_data_upload.csv')
    df_encoded = load_data('notebooks/dataset/encoded_data.csv')

    submenu = st.sidebar.selectbox('Submenu', ['Descriptive', 'Plots'])
    
    if submenu == 'Descriptive':
        with st.expander("Data-set"):
            st.dataframe(df)

        with st.expander("Class Distribution"):
            st.dataframe(df['class'].value_counts())
            st.write("Shows the count of DM positive and negative cases.")

        with st.expander("Descriptive Summary"):
            st.dataframe(df_encoded.describe())
            st.write("Statistical summary of the dataset.")

        with st.expander("Gender Distribution"):
            st.dataframe(df['Gender'].value_counts())
            st.write("Shows the count of male and female test subjects.")

    elif submenu == 'Plots':
        st.subheader("Data Visualizations")
        st.write("Visualizations help in understanding data in a simple way. Check out the following plots.")

        with st.expander("Class Distribution"):
            class_count = df_encoded['class'].value_counts().reset_index()
            class_count.columns = ['class', 'count']
            class_count['class'] = class_count['class'].map({0: 'Negative', 1: 'Positive'})

            colors = {'Positive': 'red', 'Negative': 'green'}

            fig = px.bar(class_count, x='class', y='count',
                         labels={'class': 'Class', 'count': 'Count'},
                         title='Count vs Class',
                         template="plotly_dark",
                         color='class',
                         color_discrete_map=colors,
                         text='count')

            fig.update_traces(textposition='inside')
            fig.update_layout(width=600, height=400)
            fig.update_traces(marker_line_width=2, marker_line_color='white')

            st.plotly_chart(fig)

            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)

            st.download_button(
                label="Download plot",
                data=buf,
                file_name="Count_vs_Class.png",
                mime="image/png"
            )

        with st.expander("Gender Distribution"):
            class_count = df_encoded['gender'].value_counts().reset_index()
            class_count.columns = ['gender', 'count']
            class_count['gender'] = class_count['gender'].map({0: 'Female', 1: 'Male'})

            colors = {'Male': 'orange', 'Female': 'green'}

            fig = px.bar(class_count, x='gender', y='count',
                         labels={'gender': 'Gender', 'count': 'Count'},
                         title='Count vs Gender',
                         template="plotly_dark",
                         color='gender',
                         color_discrete_map=colors,
                         text='count')

            fig.update_traces(textposition='inside')
            fig.update_layout(width=600, height=400)
            fig.update_traces(marker_line_width=2, marker_line_color='white')

            st.plotly_chart(fig)

            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)

            st.download_button(
                label="Download plot",
                data=buf,
                file_name="Count_vs_Gender.png",
                mime="image/png"
            )

        with st.expander("Distribution Plot"):
            attribute_list = ['polyuria', 'polydipsia', 'sudden_weight_loss',
                              'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
                              'itching', 'irritability', 'delayed_healing', 'partial_paresis',
                              'muscle_stiffness', 'alopecia', 'obesity']

            selected_attribute = st.selectbox("Select attribute", attribute_list)

            def distribution_plot(attribute):
                if attribute not in df_encoded.columns:
                    raise ValueError(f"Attribute '{attribute}' not found in the DataFrame.")
        
                temp_df = df_encoded.copy()
                bins = [0, 20, 40, 60, 80, 100]
                labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
                temp_df['age_group'] = pd.cut(temp_df['age'], bins=bins, labels=labels, right=False)

                temp_df['class'] = temp_df['class'].map({0: 'Negative', 1: 'Positive'})
                temp_df['gender'] = temp_df['gender'].map({0: 'Female', 1: 'Male'})

                male_df = temp_df[temp_df['gender'] == 'Male']
                female_df = temp_df[temp_df['gender'] == 'Female']

                dataframe_1 = male_df.groupby(['age_group', 'class'])[attribute].sum().reset_index(name='count')
                dataframe_2 = female_df.groupby(['age_group', 'class'])[attribute].sum().reset_index(name='count')

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Male', 'Female'),
                    shared_yaxes=True,
                    horizontal_spacing=0.1
                )

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
                fig.update_layout(width=600, height=400)
                fig.update_traces(marker_line_width=1, marker_line_color='white')

                st.plotly_chart(fig)

                buf = io.BytesIO()
                fig.write_image(buf, format="png")
                buf.seek(0)

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
