import streamlit as st

# Title and Layout:
st.set_page_config(page_title="Fake news detection using BERTopic", layout="wide")
st.title("Fake news detection using BERTopic")
st.header("""
            A simple web app to detect if a given document or a web article belongs to a common fake news topic or not.""")

st.markdown("""        
            This demo app is based on BERTopic, a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.
            
            The models were trained on Misinfo dataset from Kaggle, based on EUvsDisinfo data, using miniLM Sentence Transformer embedding. This application seeks to combine unsupervised learning techniques and labelled data to detect fake news. Two models were trained based on fake and true news datasets, and topics were generated using BERTopic. Prediction is done by modeling the topic of the input and compared to the topics clustered by the model based on the two datasets. If the input topic is similar to the topics in the fake news dataset, the input is detected to be in common fake news topics.                      
            
            At the moment, the accuracy in prediction of this demo app is not too high due to the limited hosting space and training data, but is is showing promise. Better and larger embedding models (such as paraphrase-multilingual-mpnet-base-v2) can give much better detection, but it goes over the limit of Streamlit Community Cloud. Performance can also be further improved upon by continually adding more data to the training set. 
            """)

# Some other things to write in here, such as what is BERTopic and how it works, how the models were trained, etc.
st.subheader("About BERTopic:")
st.markdown("""
            BERTopic is an impressive tool developed by Maarten Grootendorst (2022). It combines the strengths of BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. Grootendorst also built BERTopic in a modular manner, allowing easy customization and integration with other NLP tools. Grootendorst's principles is interesting because it may be possible to modify this pipeline to other types of data such as images, sounds, etc.            
            
            BERTopic can be found in this link from Grootendorst's Github: https://maartengr.github.io/BERTopic/index.html
            """)

st.subheader("About the dataset:")
st.markdown("""
            The MisInfo dataset is a 79k news dataset from Kaggle, based on EUvsDisinfo data. It contains 43642 misinfo, fake news or propaganda articles and 34975 'true' news articles. The structure of the dataset made it easy to train classification models, however topic modelling is an interesting approach towards the problem of fake news detection.
            
            The dataset can be found in this link: https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k
            """)

st.subheader("About the models:")
st.markdown("""
            The idea behind this demo app is that we can detect fake news by comparing the topic of a given document or a news article, then compare it to the topics generated by the two models trained on the two datasets. If the input topic is similar to the topics in the fake news dataset, the input is likely to be in a common fake news topic. This is by no means a perfect solution, but it can be a good way to quickly skim through a large number of documents and articles to find the ones that are most likely to be fake.
            
            The problem with this approach is that clustering can give a general idea about the given article, but it does not give a clear answer about the article's truthfulness. For example, an article about Donald Trump or Obamacare can be detected as fake news because it is a common topic in fake news, but it can still be truthful. Using this framework on a larger model such as paraphrase-multilingual-mpnet-base-v2 gave 47\% accuracy in prediction on a labelled test set, which is usable for skimming purposes. This is why it is important to combine this approach with other methods such as classification and fact-checking, etc.
            
            Moreover, common topics in fake news evolve over time, so the data and models need to be updated regularly. The dataset was published in July 2022, and so far at least we have the destruction of Nord Stream pipepline as a new topic in fake news. This is why it is important to keep updating the dataset and the models down the line to keep up with the evolution of fake news.
            
            The codes for the whole framework can be found here: https://github.com/HIEUNT1610/NLP---Fake-news-detection-using-BERTopic
            """)
