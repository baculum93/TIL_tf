import pandas as pd
import numpy as np
import scipy.stats as stats
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import text_to_word_sequence

df = pd.read_csv(".\Mo_precursor.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
abstract_txt = []
title_txt = []
for row in df.itertuples():
    if type(row.paper_title) != float and type(row.abstract) != float:
        abstract_tokens = word_tokenize(row.abstract)
        title_tokens = word_tokenize(row.paper_title)
        title_stop_tokens = [word for word in title_tokens if not word in stop_words]
        abstract_stop_tokens = [word for word in abstract_tokens if not word in stop_words]
        lemmatize_abstract = ' '.join([lemmatizer.lemmatize(w) for w in abstract_stop_tokens])
        lemmatize_title = ' '.join([lemmatizer.lemmatize(w) for w in title_stop_tokens])
        abstract_txt.append(lemmatize_abstract)
        title_txt.append(lemmatize_title)
    else: 
        abstract_txt.append('')
        title_txt.append('')

df['title_2']=title_txt
df['abstract_2']=abstract_txt
df.to_csv("./pre_Mo_precursor.csv", encoding="utf-8-sig")