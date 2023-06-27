# NLP---Fake-news-detection-using-BERTopic
This is a simple web app to predict if a given document or a web article is true or fake.
            
The app is based on BERTopic, a topic modelling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.
            
The models were trained on the Misinfo dataset from Kaggle, based on EUvsDisinfo data.
This application seeks to combine unsupervised and supervised learning techniques to detect fake news. 
            
This web app allows a rudimentary detection of fake news based on topic modelling. Two models were trained based on fake and true news datasets, and topics were generated. Prediction can be done by modelling the topic of the input and compared to the topics clustered by the model based on the two datasets. 
            
Users can simply upload pdf files or input the URL of a web article to get the prediction.
            
At the moment, the accuracy in prediction is not too high due to the limited training data, and therefore, can be further improved upon by continually adding more data to the training set. 
