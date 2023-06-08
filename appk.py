#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 
import sklearn
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("model_NB.sav", "rb"))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("fitur_baru_seleksi_tf_idf.sav", "rb"))))

# Judul Halaman
st.title ('Prediksi Berita')

clean_teks = st.text_input('Masukkan Berita')

data_deteksi = ''

if st.button('Hasil Deteksi'):
    predict_berita = model.predict(loaded_vec.fit_transform([clean_teks]))
    
    if (predict_berita==0):
        data_deteksi = "Berita Fake"
    else:
        data_deteksi = "Berita Real"
        
st.success(data_deteksi)


# In[ ]:




