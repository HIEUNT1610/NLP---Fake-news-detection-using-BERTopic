# Getting started
import streamlit as st
from Home import topic_model_fake, topic_model_true

# Title and Layout:
#st.set_page_config(page_title="Fake news detection using BERTopic", layout="wide")
st.title("Fake news detection using BERTopic")
st.header("""
            A simple web app to detect if a given document or a web article is true or fake.
            """)

# In here we are just going to show the visualizations of the topics. We will use the same models as in the Home.py file.
st.subheader("Visualizations of the trained topics:")
st.markdown("Which type of visualization would you like to see?")
option1 = st.selectbox("Select model:", ("Fake news", "True news"))
option2 = st.selectbox("Select visualization:", ("Top 30 topics", "Topic distribution", "Topic hierarchy"))

# Functions for visualizations. This is for caching the results so they don't have to be loaded each time:
@st.cache_data
def visualize_topics():
    return topic_model.visualize_topics(top_n_topics=100, custom_labels= True, width=700, height=700)
@st.cache_data
def visualize_hierarchy():
    return topic_model.visualize_hierarchy(top_n_topics=30, custom_labels=True, width=700, height=700)
@st.cache_data
def get_topic_info():
    return topic_model.get_topic_info().head(30)

# Visualizations:
if option1 and option2:
    if option1 == "Fake news":
        topic_model = topic_model_fake
    else:
        topic_model = topic_model_true

    if option2 == "Topic distribution":
        st.write(visualize_topics())
    elif option2 == "Top 30 topics":
        st.write(get_topic_info())
    else:
        st.write(visualize_hierarchy())