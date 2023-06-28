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

# Title and Layout:
st.set_page_config(page_title="Fake news detection using BERTopic", layout="wide")
st.title("Fake news detection using BERTopic")
st.header("""
            A simple web app to detect if a given document or a web article belongs to a common fake news topic or not.""")

st.markdown("""        
            This demo app is based on BERTopic, a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.
            """)
st.subheader("How to use this app:")  
st.markdown("""  
            Users can simply upload PDF files or input the URLs of web articles to detect.""")

# Function for models loading:
@st.cache_resource(ttl=3600) # Cache the model so it doesn't have to be loaded each time. Limit to 1 hour.
def download_and_cache_models():
    """Download pre-trained models from Google Drive.
    This function returns 2 BERTopic models, one trained on fake news and one trained on true news.
    The models were trained on Misinfo dataset from Kaggle, based on EUvsDisinfo data.
    Models are cached so they don't have to be downloaded each time."""
    #gdown.download(id = "1XJfCt7PFm0LlZBMDKJF-9BvukG8Pj0Yo", output = "misinfo-fake-pickle", quiet=False)
    #gdown.download(id = "1Bt7LDObSscall84N344uwkXhJIxXfsHZ", output = "misinfo-true-pickle", quiet=False)  
        
    # Load models. Loading without embedding made things worse, but it's not possible to do otherwise with streamlit sharing:
    #sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model_fake = BERTopic.load("misinfo-fake-minilm.pickle", embedding_model= sentence_model)
    topic_model_true = BERTopic.load("misinfo-true-minilm.pickle", embedding_model= sentence_model)

    return topic_model_fake, topic_model_true, sentence_model

# Functions for reading pdf files:
from io import BytesIO
def read_pdf(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    
    for page in range(len(reader.pages)):
        page_text = reader.pages[page].extract_text()
        # Remove headers and footers
        page_text = re.sub(r'\d+/\d+/\d+, \d+:\d+ [AP]M.*?https?://.*?/\d+/\d+', '', page_text)
        # Remove page numbers
        page_text = re.sub(r'Page \d+ of \d+', '', page_text)
        # Remove links
        page_text = re.sub(r'https?://\S+', '', page_text)
        
        text += page_text + "\n"

    return text

def extract_text_from_pdfs(uploaded_files):
    df = pd.DataFrame(columns=["file", "text"])
    for uploaded_file in uploaded_files:
        # Read the PDF file
        text = read_pdf(uploaded_file)
        # Add the file name and the text to the data frame
        name = uploaded_file.name
        name = re.sub('.pdf', '', name)
        df = df.append({"file": name, "text": name + ' ' + text}, ignore_index=True)
    return df


# Function for prediction:
def predict(documents, topic_model_fake, topic_model_true, sentence_model):
    """This function takes in a list of pdf files and prints out the predicted topics based on the trained models.
    documents: a dataframe with 2 columns: name and text
    topic_model_fake: the trained topic model for fake news, imported from pickle
    topic_model_true: the trained topic model for true news, imported from pickle
    sentence_model: the trained sentence transformer model"""
    
    # Vectorize the documents
    test_embeddings = sentence_model.encode(documents["text"].astype(str).tolist())
    
    # Perform topic predictions on the documents
    test_topics_true, test_probs_true = topic_model_true.transform(documents['text'].astype(str).tolist(), test_embeddings)
    test_topics_fake, test_probs_fake = topic_model_fake.transform(documents['text'].astype(str).tolist(), test_embeddings)
    
    # Print out predictions:
    documents["prediction"] = ""
    result_str = []
    for i in range(len(documents['text'].tolist())):
        # if both models cannot predict the document
        if not topic_model_true.get_topic(test_topics_true[i]) and not topic_model_fake.get_topic(test_topics_fake[i]):
            documents["prediction"][i] = "Not sure"
            result_str.append("Model could not predict reliably if the document is true or fake based on training data.\n")
        # if predicted topic in true model but not in fake model    
        elif topic_model_true.get_topic(test_topics_true[i]) and not topic_model_fake.get_topic(test_topics_fake[i]):
            documents["prediction"][i] = "Not in fake news topics"            
            result_str.append(f"In topic {topic_model_true.topic_labels_[test_topics_true[i]]} at {test_probs_true[i][test_topics_true[i]]*100:0.4f}%.\n")
        # if predicted topic in fake model but not in true model    
        elif topic_model_fake.get_topic(test_topics_fake[i]) and not topic_model_true.get_topic(test_topics_true[i]):
            documents["prediction"][i] = "In common fake news topics"
            result_str.append(f"In the topic {topic_model_fake.topic_labels_[test_topics_fake[i]]} at {test_probs_fake[i][test_topics_fake[i]]*100:0.4f}%.\n")
        # if predicted topic in both models
        else:
            if test_probs_true[i][test_topics_true[i]] > test_probs_fake[i][test_topics_fake[i]]:
                documents["prediction"][i] = "Not in fake news topics"
                result_str.append(f"In the topic {topic_model_true.topic_labels_[test_topics_true[i]]} at {test_probs_true[i][test_topics_true[i]]*100:0.4f}%.\n")
            else:
                documents["prediction"][i] = "In common fake news topics"
                result_str.append(f"In the topic {topic_model_fake.topic_labels_[test_topics_fake[i]]} at {test_probs_fake[i][test_topics_fake[i]]*100:0.4f}%.\n")
    
    return result_str


# Function for scraping webpages:
def scrape_webpages(urls):
    """Scrape the contents of a list of webpages and return a DataFrame"""
    data = []
    for url in urls:
        response = requests.get(url)
        if response.status_code != 200:
            # Attempt to bypass paywall using 12ft.io
            try:
                response = requests.get(f'https://api.12ft.io/v1/extract?url={url}')
                text = response.json().get('data', {}).get('text', '')
            except:
                text = ''
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
        name = url.split('/')[-1]
        name = re.sub("[-_.]", ' ', name)
        data.append({'file': name, 'text': name + ' ' + text})
    df = pd.DataFrame(data)
    return df

# Visualisation functions: just use BERTopic's visualisation functions
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

from requests.exceptions import MissingSchema

# Scripts for the app:
# Loading models:
with st.spinner("Loading model. Please read descriptions in the meantime..."):
    topic_model_fake, topic_model_true, sentence_model = download_and_cache_models()
documents = pd.DataFrame(columns=["file", "text"])

st.subheader("File uploader / URL scraper")
# File uploader:
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"],
                               accept_multiple_files=True)
if pdf_files: 
    with st.spinner("Processing PDF files..."): 
        documents = extract_text_from_pdfs(pdf_files)
    
# Text input for urls:
urls = st.text_input("Enter URLs, separated by commas")
if urls:
    urls = [url.strip() for url in urls.split(',')]
    try:
        with st.spinner("Scraping webpages…"):
            url_documents = scrape_webpages(urls)
    except MissingSchema as e:
        st.error(f"Invalid URL: {e}. Please enter a valid URL with a schema (e.g. http:// or https://).")
    else:
        documents = pd.concat([documents, url_documents], ignore_index=True)

# Perform prediction:
if st.button("Predict"):
    if not documents.empty:
        with st.spinner("Predicting. Please hold..."):
            st.write("Predicting if documents are in common fake news topic or not: \n")
            result_str = predict(documents, topic_model_fake, topic_model_true, sentence_model)
            documents["result"] = result_str
            documents = documents[["file", "prediction", "result"]]
            st.dataframe(documents)
    else:
        st.write("Please upload pdf files and/or enter urls.")
        
# In here we are just going to show the visualizations of the topics. We will use the same models as in the Home.py file.
st.subheader("Visualizations of the trained topics:")
st.markdown("Which type of visualization would you like to see?")
option1 = st.selectbox("Select model:", (" ", "Fake news", "True news"))
option2 = st.selectbox("Select visualization:", (" ", "Top 30 topics", "Topic distribution", "Topic hierarchy"))

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