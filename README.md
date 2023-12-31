# Fake News Detection using BERTopic

## Overview

This repository presents a simple demonstration of fake news detection using topic modelling. The demo utilizes BERTopic, a topic modelling technique that harnesses BERT embeddings and c-TF-IDF to form dense clusters, facilitating easily interpretable topics while retaining crucial words in topic descriptions.

## Purpose

The primary goal of this demo is two-fold:

1. **Detection Tool:** Offer a tool for quickly identifying potentially fake news articles, paving the way for further in-depth analysis.
  
2. **Showcase Skills:** Demonstrate proficiency in addressing a problem and developing a comprehensive solution from end to end.

## Approach

The core of the demo relies on comparing the topic of a given document or news article with the topics generated by two BERTopic models trained on fake and true news datasets. If the input document's topic aligns with the topics from the fake news dataset, it raises a flag for potential falsity, urging a more thorough verification process. While not a flawless solution, this method provides a rapid means to sift through numerous documents and articles, pinpointing those most likely to be fake.

## Model Details

The BERTopic models utilized in this demo were trained on the Misinfo dataset from Kaggle, derived from EUvsDisinfo data, using miniLM Sentence Transformer embedding. The combination of unsupervised learning techniques and labelled data enables effective fake news detection. The prediction process involves generating the input topic and comparing it to the topics clustered by the models based on the two datasets.

## Versions

Two versions of the demo are available:

1. **Streamlit Version (Small):** Yields reasonable results, though the potential for improvement exists due to limited hosting space on the Streamlit Community Cloud.

2. **Main Version (Large):** Provides superior and more easily interpreted topics, showing promise. Enhanced detection capabilities can be achieved with larger embedding models (e.g., paraphrase-multilingual-mpnet-base-v2). Continuous performance enhancement is possible by augmenting the training set with more data. A deployment on Google Cloud Service for the larger version will be accessible soon.

## How to Use

1. **Clone the Repository:**
   ```
   git clone https://github.com/HIEUNT1610/NLP---Fake-news-detection-using-BERTopic.git
   cd NLP---Fake-news-detection-using-BERTopic
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Demo:**
   ```
   python -m streamlit run Home.py
   ```

This command will initiate the interactive interface, allowing users to explore and test the fake news detection capabilities.
