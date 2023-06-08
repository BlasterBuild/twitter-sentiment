import streamlit as st
import numpy as np
import pickle
import nltk
import contractions, string
import pandas
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from PIL import Image
import time





# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_tfidf = pickle.load(open('tfidf_v.sav', 'rb'))
st.title("Product review sentiment analysis")
input_rev = st.text_area("", height=10, placeholder='write your review here')




def space_contractions(text):
    text = contractions.fix(text)
    return text


punc = string.punctuation


def remove_punctuation(text):
    for char in punc:
        text = text.replace(char, ' ')
    return text


stop_words = stopwords.words('english')


def remove_stopwords(data):
    swords = []
    for stopword in data.split():
        if stopword in stop_words:
            swords.append('')
        else:
            swords.append(stopword)
    return " ".join(swords)


list = []
stemmer = PorterStemmer()


def stemming(text):
    for word in text.split():
        list.append(stemmer.stem(word))
    return " ".join(list)


if st.button("Predict", use_container_width=True):
    prgress_text = "Operation in progress please wait"
    my_bar = st.progress(0, text=prgress_text)
    time.sleep(0.1)
    my_bar.progress(50, text=prgress_text)
    time.sleep(0.1)
    my_bar.progress(100, text=prgress_text)
    # preprocessing
    # contractions
    input_rev = space_contractions(input_rev)
    # lowercase
    input_rev = input_rev.lower()
    # removing punctuations
    input_rev = remove_punctuation(input_rev)
    # remove stopwords
    input_rev = remove_stopwords(input_rev)
    # stemming
    input_rev = stemming(input_rev)
    input_rev = [input_rev]

    # vectorize

    tfidf_vectorize = loaded_tfidf.transform(input_rev)

    # predict

    result = loaded_model.predict(tfidf_vectorize)


# display





if (result == 1):
    st.header(':blue[Positive] :sunglasses:')
    st.balloons()


    def check():
        return True
else:
    st.header(':red[Negative]')
    st.snow()
