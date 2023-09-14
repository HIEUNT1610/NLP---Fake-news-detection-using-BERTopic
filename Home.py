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
import torch
import matplotlib.pyplot as plt

# Title and Layout:
st.set_page_config(page_title="Fake news detection using BERTopic", layout="wide")
st.title("Fake news detection using BERTopic")
st.header("""
            A simple demo to detect if a given document or a web article belongs to a common fake news topic or not.""")

st.markdown("""        
            This demo app is based on BERTopic, a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. The topics for the input documents are generated based on the content using a combination of topic modelling techniques, and are compared to the topics of fake news and true news from the MisInfo Kaggle dataset. The detection is based on the similarity between the input document and the fake news topics, using a simple neural network trained on the ground truth provided by the dataset. 
            
            This demo app's purpose is to: (i) provide a quick detection tool for further analysis; (ii) demonstrate my ability in looking at a problem and creating a solution from end to end. For more details, please refer to the About page.
            """)
st.subheader("How to use this app:")  
st.markdown("""  
            Users can simply upload PDF files or input the URLs of web articles to start detection.
            Please ensure that URLs are not behind paywalls, or the app will not be able to access the content and will not be able to accurately model the topic of the documents.
            """)

# Fake Detect model:
# Initalize a MLP for binary classification using PyTorch with 1 hidden layer of 100 neurons
class FakeDetect(torch.nn.Module):
    def __init__(self, early_stop = False, epoch = 50, batch_size = 32):
        super(FakeDetect, self).__init__()
        input_dim = 768 #2 times of the embedding dimension
        hidden_dim = 100
        output_dim = 2
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim) #[768, 100]
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim) #[100, 2]
        self.sigmoid = torch.nn.Sigmoid() # Sigmoid for binary classification
        self.activation = torch.nn.ReLU()
        self.early_stop = early_stop
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

        
    def forward(self, x):
        out = self.linear1(x) # [batch_size, X.shape[1]] . [X.shape[1], 100] = [batch_size, 100]
        out = self.activation(out) # [batch_size, 100]
        out = self.linear2(out) # [batch_size, 100] . [100, 2] = [batch_size, 2]
        out = self.sigmoid(out) # [batch_size, 2]
        return out
    
    def fit(self, X_train, y_train, X_val, y_val):
        """ 
        This function trains the model.
        It takes as inputs the training data and the validation data, and returns the accuracy of the model on validation set.
        """
        train_losses = []
        for epoch in range(self.epoch):
            print('Epoch:', epoch)
            epoch_loss = 0
            i = 0
            while i < len(X_train):
                X_batch = X_train[i:i+32]
                y_batch = y_train[i:i+32]
                i += 32
                # Zero the gradients
                optimizer = self.optimizer
                optimizer.zero_grad()
                # Forward pass
                output = self.forward(X_batch)
                # Calculate the loss
                loss = self.loss_function(output, y_batch.long())
                epoch_loss += loss.item()
                loss.backward()
                # Update the weights
                optimizer.step()
            print("Loss on training set at epoch %d : %f" %(epoch, epoch_loss))
            train_losses.append(epoch_loss)
        # Validation
            with torch.no_grad():
                if self.early_stop:
                    output = self.forward(X_val)
                    val_loss = self.loss_function(output, y_val.long())
                    print("Loss on validation set at epoch %d : %f" %(epoch, val_loss))
                    # prediction and accuracy on the dev set
                    pred_labels = torch.argmax(output, dim=1)
                    accuracy = torch.sum(pred_labels == y_val).item() / len(y_val)
                    print("Accuracy on validation set, after epoch %d: %3.2f\n" % (epoch, accuracy * 100))
                    # early stopping
                    # if first epoch: we record the dev loss, to be used for early stopping
                    if epoch == 0:
                        previous_val_loss = val_loss
                    elif val_loss > previous_val_loss:
                        print("Loss on validation set has increased, we stop training!")
                        break
                    else:
                        previous_val_loss = val_loss
        # Plot the training losses
        plt.plot(train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training losses over Epochs")
        plt.show()    
    
    def transform(self, X, y):
        """
        This function evaluates the model on an annotated test set, and returns the accuracy on that set.
        """
        with torch.no_grad():
            log_probs = self.forward(X)
            pred_labels = torch.argmax(log_probs, dim=1)
            test_loss = self.loss_function(log_probs, y.long())
            accuracy = torch.sum(pred_labels == y).item() / len(y)
            # Print the results
            print("Result of the original model:")
            print("Loss on test set after training: %f\n" %(test_loss))
            print("Accuracy on test set after training: %3.2f\n" % (accuracy * 100))
    
    def predict(self, X):
        """
        This function predicts the labels of a set of documents, and returns a list of predicted labels.
        """
        with torch.no_grad():
            log_probs = self.forward(X)
            pred_labels = torch.argmax(log_probs, dim=1)
            return pred_labels.tolist()
        
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
    fake_detector = FakeDetect(early_stop=True)
    fake_detector.load_state_dict(torch.load("fakenews_classif.pth"))
    
    return topic_model_fake, topic_model_true, sentence_model, fake_detector

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
    
    # Concatenate the topic embeddings and the document embeddings:
    topic_embeddings = np.zeros((len(documents["text"].tolist()), 768))
    zeros = np.zeros((384))
    for i in range(len(test_embeddings)):
        if not topic_model_true.get_topic(test_topics_true[i]) and not topic_model_fake.get_topic(test_topics_fake[i]):
            topic_embeddings[i] = np.concatenate((test_embeddings[i], zeros))
        elif topic_model_true.get_topic(test_topics_true[i]) and not topic_model_fake.get_topic(test_topics_fake[i]):
            topic_embeddings[i] = np.concatenate((test_embeddings[i], topic_model_true.topic_embeddings_[test_topics_true[i]]))
        elif topic_model_fake.get_topic(test_topics_fake[i]) and not topic_model_true.get_topic(test_topics_true[i]):
            topic_embeddings[i] = np.concatenate((test_embeddings[i], topic_model_fake.topic_embeddings_[test_topics_fake[i]]))
        else:
            if test_probs_true[i][test_topics_true[i]] > test_probs_fake[i][test_topics_fake[i]]:
                topic_embeddings[i] = np.concatenate((test_embeddings[i], topic_model_true.topic_embeddings_[test_topics_true[i]]))
            else:
                topic_embeddings[i] = np.concatenate((test_embeddings[i], topic_model_fake.topic_embeddings_[test_topics_fake[i]]))
                
    # Print out predictions:
    #TODO: Something with the topic get messed up after the MLP, so need to think of someway to access the topic labels.
    pred_labels = fake_detector.predict(torch.Tensor(topic_embeddings))
    documents["prediction"] = ""
    result_str = []
    for i in range(len(topic_embeddings)):
        if pred_labels[i] == 0:
            documents["prediction"][i] = "Likely in common fake news topics"
            if topic_model_fake.get_topic(test_topics_fake[i]):
                result_str.append(f"Document is likely in the topic {topic_model_fake.topic_labels_[test_topics_fake[i]]}")
            else:
                result_str.append(f"Document is likely in the topic {topic_model_true.topic_labels_[test_topics_true[i]]}")

        else:
            documents["prediction"][i] = "Not in fake news topics"
            if topic_model_fake.get_topic(test_topics_fake[i]):
                result_str.append(f"Document is likely in the topic {topic_model_fake.topic_labels_[test_topics_fake[i]]}")
            else:
                result_str.append(f"Document is likely in the topic {topic_model_true.topic_labels_[test_topics_true[i]]}")
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
    topic_model_fake, topic_model_true, sentence_model, fake_detector = download_and_cache_models()
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
        with st.spinner("Detecting. Please hold..."):
            st.write("Detecting if documents are in common fake news topic or not: \n")
            result_str = predict(documents, topic_model_fake, topic_model_true, sentence_model)
            documents["result"] = result_str
            documents = documents[["file", "prediction", "result"]]
            st.dataframe(documents, use_container_width=True)
    else:
        st.write("Please upload pdf files and/or enter urls.")
        
# In here we are just going to show the visualizations of the topics. We will use the same models as in the Home.py file.
st.subheader("Visualizations of the trained topics:")
st.markdown("In this section you can see the visualizations of the trained topics. The topics are generated from the fake news dataset and the true news dataset. The visualizations are made with the BERTopic package, and you can check Top 30 topics, Topic distribution, and Topic hierarchy.")
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