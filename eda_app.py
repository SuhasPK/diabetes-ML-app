import streamlit as st # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import seaborn as sns # type: ignore
import plotly.express as px # type: ignore
import io

def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader('Exploratory Data Analysis')
    df = load_data('data/diabetes_data_upload.csv')
    df_encoded = load_data('data/diabetes_data_upload_clean.csv')
    freq_df = load_data('data/freqdist_of_age_data.csv')

    submenu = st.sidebar.selectbox('Submenu',['Descriptive','Plots'])
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
            st.write("""This show how many of the test subjects were male and female.""")

######################PLOTS###################################################

    elif submenu == 'Plots':
        st.subheader("Data visualizations")
        st.write(""" Visualization are very powerful tool for understanding data in a simple way. Do go through all of it. """)
        col1, col2 = st.columns([2,1])
        
        with col1:
            with st.expander("Gender Distribution "):
                fig, ax = plt.subplots()
    # Plotting the count plot with hue for different colors
                sns.countplot(x='Gender', data=df, ax=ax, palette='Set2')

    # Adding count labels on top of each bar
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                    (   p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')

                ax.set_title('Gender Distribution')    
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)

            # Adding download button
                st.download_button(
                    label="Download plot",
                    data=buf,
                    file_name="gender_distribution_plot.png",
                    mime="image/png"
                )

            with st.expander("Class Distribution "):
                fig, ax = plt.subplots()
    # Plotting the count plot with hue for different colors
                sns.countplot(x='class', data=df, ax=ax, palette='Set2')

    # Adding count labels on top of each bar
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                    (   p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')

                ax.set_title('Class Distribution')    
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)

            # Adding download button
                st.download_button(
                    label="Download plot",
                    data=buf,
                    file_name="class_distribution_plot.png",
                    mime="image/png"
                )

            with st.expander(" 'Age' Frequency Distribution "):
                fig, ax = plt.subplots()
    # Plotting the count plot with hue for different colors
                sns.barplot(x='Age', y='count' ,data=freq_df, ax=ax, palette='Set2')

    # Adding count labels on top of each bar
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                    (   p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')

                ax.set_title('Age Frequency Distribution') 
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')   
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)

            # Adding download button
                st.download_button(
                    label="Download plot",
                    data=buf,
                    file_name="age_freq_distribution_plot.png",
                    mime="image/png"
                )

        with col2:
            with st.expander("Gender Distribution Table"):
                gen_df = df['Gender'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender type","Counts"]
                st.dataframe(gen_df)

            with st.expander("Class Distribution Table"):
                class_df = df['class'].value_counts().to_frame()
                class_df = class_df.reset_index()
                class_df.columns = ["Class type","Counts"]
                st.dataframe(class_df)

            with st.expander(" 'Age' Frequency Distribution"):
                st.dataframe(freq_df[['Age','count']])

        with st.expander("Outlier Detection Plot"):
    # Creating the box plot with grid and title
            p = px.box(df, x='Age', color = 'Gender', template='plotly_dark', title='Outlier Detection in Age Distribution')
    
    # Updating layout to include grid lines
            p.update_layout(
                xaxis=dict(showgrid=True),  # Enable grid for x-axis
                yaxis=dict(showgrid=True),  # Enable grid for y-axis
            )   
    
    # Display the plot in Streamlit
            st.plotly_chart(p)

    # Save the plot to a BytesIO object
            buf = io.BytesIO()
            p.write_image(buf, format="png")
            buf.seek(0)

    # Adding download button
            st.download_button(
                label="Download plot",
                data=buf,
                file_name="outlier_detection_plot.png",
                mime="image/png"
            )      

        with st.expander("Correlation plot"):
            corr_matrix = df_encoded.corr()
            fig = plt.figure(figsize=(25,15))
            sns.heatmap(corr_matrix, annot=True)
            plt.title("Correlation plot of UCI diabetes data")
    
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            # Adding download button
            st.download_button(
                label="Download plot",
                data=buf,
                file_name="correlation_plot.png",
                mime="image/png"
            )