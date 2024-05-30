import streamlit as st 
from joblib import load
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer
from PIL import Image
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

# Load the model
model = load('model.pkl')

st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message')

option = st.selectbox("You Got Message From:", ["Via Email", "Via SMS", "Other"])

if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = model.named_steps['tfidfvectorizer'].transform([transform_sms])
    result = model.named_steps['multinomialnb'].predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
