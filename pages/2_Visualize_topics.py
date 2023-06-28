# Getting started
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import gdown
import streamlit as st
from pypdf import PdfReader
import re
import os
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from Home import topic_model_fake, topic_model_true, sentence_model

# Title and Layout:
#st.set_page_config(page_title="Fake news detection using BERTopic", layout="wide")
st.title("Fake news detection using BERTopic")
st.header("""
            A simple web app to detect if a given document or a web article is true or fake.""")

# In here we are just going to show the visualizations of the topics. We will use the same models as in the Home.py file.
st.subheader("Visualizations of the trained topics:")
st.markdown("Which type of visualization would you like to see?")
option1 = st.selectbox("Select model:", ("Fake news", "True news"))
option2 = st.selectbox("Select visualization:", ("Top 30 topics", "Topic distribution", "Topic hierarchy"))

# Visualizations:
if option1 == "Fake news":
    topic_model = topic_model_fake
else:
    topic_model = topic_model_true

if option2 == "Topic distribution":
    st.write(topic_model.visualize_topics(top_n_topics=100, custom_labels= True, width=700, height=700))
elif option2 == "Top 30 topics":
    st.write(topic_model.get_topic_info().head(30))
else:
    st.write(topic_model.visualize_hierarchy(top_n_topics=30, custom_labels=True, width=700, height=700))


