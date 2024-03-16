import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()




def text_transform(Input_Email):
    y = []
    # STEP 1 CONVERT INTO LOWER CASE
    Input_Email = Input_Email.lower()
    # NOW STEP 2 MAKE THE TOKEN
    Input_Email = nltk.word_tokenize(Input_Email)
    # now step 3 removing the special characters
    for i in Input_Email:
        if i.isalnum():
            y.append(i)
    # Input_Email=y we never assing the list like this because it clear all the input email
    Input_Email = y[:]
    y.clear()
    # step 4 removing the stopword and puncuations
    for i in Input_Email:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # clearing the y
    Input_Email = y[:]
    y.clear()
    # step 5 stermming
    for i in Input_Email:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorized.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = text_transform(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam SMS / E-Mail")
    else:
        st.header("Not Spam")

