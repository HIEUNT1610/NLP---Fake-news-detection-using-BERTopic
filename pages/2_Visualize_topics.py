# Getting started
import streamlit as st
from Home import topic_model_fake, topic_model_true

# Title and Layout:
#st.set_page_config(page_title="Fake news detection using BERTopic", layout="wide")
st.title("Fake news detection using BERTopic")
st.header("""
            A simple web app to detect if a given document or a web article belongs to a common fake news topic or not.""")


# In here we are just going to show the visualizations of the topics. We will use the same models as in the Home.py file.
st.subheader("Visualizations of the trained topics:")
st.markdown("Which type of visualization would you like to see?")
option1 = st.selectbox("Select model:", (" ", "Fake news", "True news"))
option2 = st.selectbox("Select visualization:", (" ", "Top 30 topics", "Topic distribution", "Topic hierarchy"))

# Functions for visualizations. This is for caching the results so they don't have to be loaded each time:
@st.cache_data
def visualize_topics_fake():
    return topic_model_fake.visualize_topics(top_n_topics=100, custom_labels= True, width=700, height=700)
@st.cache_data
def visualize_topics_true():
    return topic_model_true.visualize_topics(top_n_topics=100, custom_labels= True, width=700, height=700)
@st.cache_data
def visualize_hierarchy_fake():
    return topic_model_fake.visualize_hierarchy(top_n_topics=30, custom_labels=True, width=700, height=700)
@st.cache_data
def visualize_hierarchy_true():
    return topic_model_true.visualize_hierarchy(top_n_topics=30, custom_labels=True, width=700, height=700)
@st.cache_data
def get_topic_info_fake():
    return topic_model_fake.get_topic_info().head(30)
@st.cache_data
def get_topic_info_true():
    return topic_model_true.get_topic_info().head(30)

# Visualizations:
if option1 == "Fake news" and option2 == "Topic distribution":
    st.write(visualize_topics_fake())
elif option1 == "Fake news" and option2 == "Top 30 topics":
    st.write(get_topic_info_fake())
elif option1 == "Fake news" and option2 == "Topic hierarchy":
    st.write(visualize_hierarchy_fake())
elif option1 == "True news" and option2 == "Topic distribution":
    st.write(visualize_topics_true())
elif option1 == "True news" and option2 == "Top 30 topics":
    st.write(get_topic_info_true())
elif option1 == "True news" and option2 == "Topic hierarchy":
    st.write(visualize_hierarchy_true())
else:
    st.write("Please select a model and a visualization from the dropdown menus above.")