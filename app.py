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
            A simple web app to detect if a given document or a web article is true or fake.""")

st.markdown("""        
            The app is based on BERTopic, a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.
            
            The models were trained on Misinfo dataset from Kaggle, based on EUvsDisinfo data.
            This application seeks to combine unsupervised and supervised learning techniques to detect fake news. Two models were trained based on fake and true news datasets, and topics were generated. Prediction can be done by modeling the topic of the input and compared to the topics clustered by the model based on the two datasets. If the input topic is similar to the topics in the fake news dataset, the input is predicted to be fake, and vice versa.                      
            
            At the moment, the accuracy in prediction is not too high due to the limited training data, but is is showing promise. Therefore, it can be further improved upon by continually adding more data to the training set. 
            """)

# Function for models loading:
@st.cache_resource # Cache the model so it doesn't have to be loaded each time
def download_and_cache_models():
    """Download pre-trained models from Google Drive.
    This function returns 2 BERTopic models, one trained on fake news and one trained on true news.
    The models were trained on Misinfo dataset from Kaggle, based on EUvsDisinfo data.
    Models are cached so they don't have to be downloaded each time."""
    #gdown.download(id = "1XJfCt7PFm0LlZBMDKJF-9BvukG8Pj0Yo", output = "misinfo-fake-pickle", quiet=False)
    #gdown.download(id = "1Bt7LDObSscall84N344uwkXhJIxXfsHZ", output = "misinfo-true-pickle", quiet=False)  
        
    # Load models. Loading without embedding made things worse, but it's not possible to do otherwise with streamlit sharing:
    sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    topic_model_fake = BERTopic.load("misinfo-fake-model.pickle", embedding_model= sentence_model)
    topic_model_true = BERTopic.load("misinfo-true-model.pickle", embedding_model= sentence_model)

    return topic_model_fake, topic_model_true, sentence_model

# Funny loading animation:
def start_app():
    st.subheader("How to use this app:")  
    st.markdown("""  
            Users can simply upload pdf files or input the url of a web article to get the prediction.""")
    with st.spinner("Loading model. Please read descriptions in the meantime..."):
        topic_model_fake, topic_model_true, sentence_model = download_and_cache_models()
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
            documents["prediction"][i] = "True"            
            result_str.append(f"In topic {topic_model_true.topic_labels_[test_topics_true[i]]} at {test_probs_true[i][test_topics_true[i]]*100:0.4f}%.\n")
        # if predicted topic in fake model but not in true model    
        elif topic_model_fake.get_topic(test_topics_fake[i]) and not topic_model_true.get_topic(test_topics_true[i]):
            documents["prediction"][i] = "Fake"
            result_str.append(f"In the topic {topic_model_fake.topic_labels_[test_topics_fake[i]]} at {test_probs_fake[i][test_topics_fake[i]]*100:0.4f}%.\n")
        # if predicted topic in both models
        else:
            if test_probs_true[i][test_topics_true[i]] > test_probs_fake[i][test_topics_fake[i]]:
                documents["prediction"][i] = "True"
                result_str.append(f"In the topic {topic_model_true.topic_labels_[test_topics_true[i]]} at {test_probs_true[i][test_topics_true[i]]*100:0.4f}%.\n")
            else:
                documents["prediction"][i] = "Fake"
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


from requests.exceptions import MissingSchema

# Scripts for the app:
# Loading models:
topic_model_fake, topic_model_true, sentence_model = start_app()
documents = pd.DataFrame(columns=["file", "text"])

st.subheader("File uploader / URL scraper")
# File uploader:
pdf_files = st.file_uploader("Upload pdf files", type=["pdf"],
                               accept_multiple_files=True)
if pdf_files: 
    with st.spinner("Processing pdf…"): 
        documents = extract_text_from_pdfs(pdf_files)
    
# Text input for urls:
urls = st.text_input("Enter urls separated by commas")
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
            st.write("Predicting true or fake: \n")
            result_str = predict(documents, topic_model_fake, topic_model_true, sentence_model)
            documents["result"] = result_str
            documents = documents[["file", "prediction", "result"]]
            st.dataframe(documents)
    else:
        st.write("Please upload pdf files and/or enter urls.")