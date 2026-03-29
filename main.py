#GPT
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download once
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

import streamlit as st
import  pickle
import string

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
#GPT
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = text.split()

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
tfidf = pickle.load(open('vectorizer.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the Mail:")
st.write(model)

if st.button("Predict"):

    if input_sms.strip() != "":

        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # predict
        result = model.predict(vector_input)[0]

        # display
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

    else:
        st.warning("⚠️ Please enter a message first")



