# Video to Audio converter
import os
command2mp3 = os.system('ffmpeg -i C:\\a_work\\a_iot\\video2audio\\video.MP4 C:\\a_work\\a_iot\\video2audio\\video.wav')

#create chunks of Audio file for better conversion
from pydub import AudioSegment
from pydub.utils import make_chunks
myaudio = AudioSegment.from_file("C:\\a_work\\a_iot\\video2audio\\video.wav" , "wav")
chunk_length_ms = 70000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files
for i, chunk in enumerate(chunks):
    # chunk_name = "data\_{0}.wav".format(i)#{0}_{1:0>{2}}
    chunk_name = "C:\\a_work\\a_iot\\video2audio\\data\\chunk{:05}.wav".format(i)
    # chunk_name = "data\chunk{0}.wav".format(i)
    print("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")
# ###########################----audio to text conversion---####################################
def audio2text(src):
    import speech_recognition as sr
    # fp = open(path, 'rb')
    print('source of data folder',src)
    r = sr.Recognizer()

    # audio = 'LongWelcome.wav'
    audio = src
    # audio = 'C:\\Users\\u341010\\Downloads\\LongWelcome.wav'
    with sr.AudioFile(audio) as source:
        audio = r.record(source)
        # audio = r.record(source, duration=100)
        print('Done!')

    try:
        text = r.recognize_google(audio)
        # print(text)
        return text
    except Exception as e:
        print(e)
#-----------------------end of the audio to text function--------------------
from app import *
# print(len([iq for iq in os.scandir('C:\\a_work\\a_iot\\video2audio\\data')]))
count_files = len([iq for iq in os.scandir('C:\\a_work\\a_iot\\video2audio\\data')])
# os.chdir('C:\\a_work\\a_iot\\video2audio\\data')
MAX_FILE_NUM = count_files - 1  # change
fmat = lambda x: '{:05}.wav'.format(x)
print(fmat)
for i in range(0, MAX_FILE_NUM + 1):
    path = 'chunk' + fmat(i)
    print(path)
    # new_file = open(path, 'r')
    content_summary = audio2text(path)
    r = (content_summary)
    results.append(r)
# print(results)
text1 = ''.join(results)
print(text1)#
file = open("copy.txt", "w")
file.write(text1)
file.close()
#--------------------------create dataframe---------------------------------
transcripts_df = pd.DataFrame(results, columns=['content_summary'])
print(transcripts_df)
# df_pdf['metadata_storage_name'] = df_pdf['metadata_storage_path'].apply(lambda path: Path(path).name)
# print(df_pdf.head())
#--------------------------text summarization---------------------------------
import spacy
from heapq import nlargest
# Text Preprocessing Pkg
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import string
# Build a List of Stopwords
stopwords = list(STOP_WORDS)
nlp = spacy.load("en_core_web_sm")
# Build an NLP Object
doc = nlp(text1)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
mysentences=[sents for sents in doc.sents]
print(mysentences)
total_sentences=len(mysentences)
num_sent=int(total_sentences*0.3)
print('number-------', num_sent)

punctuation=string.punctuation
punctuations=punctuation+'\n'
sentences_frequencies = {}
word_frequencies={}

# def word_frequency(doc):
#     mytokens = [token.text.lower().strip() for token in doc if token.text not in punctuations]
#     mytokens = [token for token in mytokens if token not in stopwords]
#     return mytokens
#
# def calc_word_frequency(temp):
#     for word in temp:
#         if word not in word_frequencies.keys():
#             word_frequencies[word] = 1
#         else:
#             word_frequencies[word] += 1

for sent in mysentences:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentences_frequencies.keys():
                sentences_frequencies[sent] = word_frequencies[word.text.lower()]
            else:
                sentences_frequencies[sent] += word_frequencies[word.text.lower()]

# print(sentences_frequencies)
summary=nlargest(num_sent, sentences_frequencies, key=sentences_frequencies.get)
print(summary)
#--------------------------------text summarizer function---------------------
def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency
# word.text is tokenization in spacy
    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # Calculate Sentence Score and Ranking
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    # Find N Largest
    summary_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summary_sentences ]
    summary = ' '.join(final_sentences)
    print("Original Document\n")
    print(raw_docx)
    print("Total Length:",len(raw_docx))
    print('\n\nSummarized Document\n')
    print(summary)
    print("Total Length:",len(summary))
text_summarizer(text1)
#----------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# # transcripts_df = pd.DataFrame(results)
# from time import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words="english",
#                         use_idf=True,
#                         ngram_range=(1,1), # considering only 1-grams
#                         min_df = 0.05,     # cut words present in less than 5% of documents
#                         max_df = 0.3)      # cut words present in more than 30% of documents
# t0 = time()
#
# tfidf = vectorizer.fit_transform(transcripts_df)
#
# from sklearn.decomposition import NMF
#
# n_topics = 10
# nmf = NMF(n_components=n_topics,random_state=0)
#
# topics = nmf.fit_transform(tfidf)
# top_n_words = 5
# t_words, word_strengths = {}, {}
# for t_id, t in enumerate(nmf.components_):
#     t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
#     word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
# print(t_words)
# ########################################PDF file DataFrame df_pdf Creation########################################
# df_pdf = pd.DataFrame(results, columns=['metadata_storage_path', 'content_summary'])
# df_pdf['metadata_storage_name'] = df_pdf['metadata_storage_path'].apply(lambda path: Path(path).name)
# print(df_pdf.head())
# df_pdf = pd.DataFrame(results)
# print(df_pdf)
# with open('outfile.txt', 'wb') as outfile:
#     outfile.write(df_pdf)
# import glob, os
# results = []
# from pathlib import Path
# os.chdir('C:\\a_work\\a_iot\\video2audio\\data')
# path = glob.glob("**/*", recursive=True)
# print(path)
# # files = glob.glob(path)
# for name in path:
#     with open(name) as f:
#             for line in f:
#                 print(line.split())
