import streamlit as st
import pickle

# Load the model
tfidf = pickle.load(open('vectorization.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title('Spam Text Classifier')

# Sidebar with Information
st.sidebar.header('About')
st.sidebar.text('This is a simple spam text classifier.')

# Input
input_text = st.text_area('Enter the Message')

# Text Processing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
ps = PorterStemmer()

def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

transformed_sms = text_preprocessing(input_text)

# Vectorization
vectorized_sms = tfidf.transform([transformed_sms])

# Prediction
output = model.predict(vectorized_sms)

# Display Prediction Result
if st.button('Check for Spam/Not Spam'):
    st.subheader('Prediction:')
    if output == 1:
        st.error('This message is Spam.')
    else:
        st.success('This message is Not Spam.')
