# Video2SpeechText-Analytics-Generator
---

Video2SpeechText Analytics Generator, NLP and Artificial Intelligence 
Human video to speech text recognition is talking about Natural language processing, speech recognition, text generation etc. In this article, we will discuss how we can convert video or audio files to text generation and developed NLP and Artificial Intelligence.
Natural Language Processing (NLP)
NLP is the ability of machines to understand and analyse human language. It is a part of the Artificial Intelligence (AI) domain with a significant overlap with Linguistics.
However, natural languages are extremely complex systems. Think of a human body; it consists of 11 separate systems (e.g., nervous, digestive etc.) working in tandem with each other. Similarly, a human language has several sub-systems such as phonology, morphology, and semantics working seamlessly with each other.
Pre-requisites:
>> Python 3.6.7
>> ffmpeg : video to audio conversion app
>> Libraries: os and speech_recognition, pydub
Step 1: Prepare directory
 Create a video2audio folder and add Microsoft Teams video files. For instance, I have created a folder ' data' and in this folder(in .mp4 format).
Step 2: libraries requirement.txt
 Import the required libraries, refer to below code:
 import os, pydub, spacy
import speech_recognition as sr
also install en_core_web_sm from spacy
python -m spacy download en_core_web_sm
Step : Software for video conversion
 I am using ffmpeg to convert the video file to audio. First, I will convert this to *.wav format, as wav format allows you to extract better features.
 Download the ffmpeg file from below link.
https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-2020-11-11-git-89429cf2f2-full_build.7z
ffmpeg-2020-11-11-git-89429cf2f2-full_build
Let's save the ffmpeg executable file into the same directory of python.py them in variables as below.
Step 4: Execute video conversion commands Let us now execute these commands using the ' os' library as below
# Video to Audio converter
import os
video2wav = os.system('ffmpeg -i C:\\a_work\\a_iot\\video2audio\\video.MP4 C:\\a_work\\a_iot\\video2audio\\video.wav')
Here, my video file name is video.mp4, convert this to video.wav then divide this wav file into multiple chunks of chunks00001.wav based on milliseconds division.
pydub calculates in millisec and initially made it for 70000 seconds duration for smooth text conversion. You can change the same as per your convenience.
 with audio as source:
chunk_length_ms = 70000 # pydub calculates in milliseconds
#create chunks of Audio file for better conversion
from pydub import AudioSegment
from pydub.utils import make_chunks
myaudio = AudioSegment.from_file("C:\\a_work\\a_iot\\video2audio\\video.wav" , "wav")
chunk_length_ms = 70000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
#Export all of the individual chunks as wav files
for i, chunk in enumerate(chunks):
    chunk_name = "C:\\a_work\\a_iot\\video2audio\\data\\chunk{:05}.wav".format(i)
    # chunk_name = "data\chunk{0}.wav".format(i)
    print("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")
outputStep 5: Load the wav file
 Now, let us load the wav file that was created in the above step.
 The below audio2text function for the audio to text conversion can be used.
#--------------------audio to text conversion----------------------
def audio2text(src):
    import speech_recognition as sr
    # fp = open(path, 'rb')
    print('source of data folder',src)
    r = sr.Recognizer()
# audio = 'LongWelcome.wav'
    audio = src
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
Read multiple chunks.wav files in for loop
import os
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
Step 6: Spit out the text file
 Lastly, as per the required, set the duration of the audio you want for further processing.
file = open("C:\\a_work\\a_iot\\video2audio\\copy.txt", "w")
file.write(text1)
file.close()
or you can create pandas data frame
import pandas
#-------------load the dataset-------------------
dataset = pandas.read_csv('C:\\a_work\\a_iot\\video2audio\\copy.txt',delimiter="\t",encoding='cp1252')
dataset.head()
Step 7: NLP Process Flow:
 The text generated can be used further for Natural language understanding with Spacy library.
A typical NLP process flow has the following steps:
1) Data Collection: Data mining or ETL (extract-transform-load) process to collect a corpus of unstructured data.
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
Step 8: Feature Engineering:
Word Embeddings: Transforming text into a meaningful vector or array of numbers.

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
newText =''
for word in doc:
    if word.pos_ in ['ADJ', 'NOUN']:
        newText = " ".join((newText, word.text.lower()))
wordcloud = WordCloud(stopwords=STOPWORDS).generate(newText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
use some STOPWORD to filter the text
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["interview", "years", "lot", "question", "people"])
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(newText)
# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
outputN-grams : An unigram is a set of individual words within a document; bi-gram is a set of 2 adjacent words within a document.

from nltk import ngrams
n = 6
sixgrams = ngrams(newText.split(), n)
for grams in sixgrams:
    print(grams)
outputTF-IDF values: Term-Frequency-Inverse-Document-Frequency is a numerical statistic representing how important a word is to a document within a collection of documents.

def noun_notnoun(phrase):
    doc = nlp(phrase) # create spacy object
    token_not_noun = []
    notnoun_noun_list = []
for item in doc:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            token_not_noun.append(item.text)
        if item.pos_ == "NOUN":
            noun = item.text
for notnoun in token_not_noun:
        notnoun_noun_list.append(notnoun + " " + noun)
return notnoun_noun_list
# noun_notnoun(newText)
for phrase in newText:
    print(noun_notnoun(newText))
2) Data Preprocessing:
Tokenization: Segmentation of running text into words.
Lemmatization: Removal of inflectional endings to return the base form.
Parts-of-speech tagging: Identification of words as nouns, verbs, adjectives etc.

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
Named-Entity-Recognition: A named entity is a "real-world object" that's assigned a name - for example, a person, a country, a product or a book title. spaCy can recognize various types of named entities in a document, by asking the model for a prediction. Because models are statistical and strongly depend on the examples they were trained on, this doesn't always work perfectly and might need some tuning later, depending on your use case.
Named entities are available as the ents property of a Doc:

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
outputUsing spaCy's built-in displaCy visualizer, here's what our example sentence and its named entities look like:
from spacy import displacy
svg1 = displacy.render(doc,style='ent',jupyter=True)
svg1
outputStep 10: Further enhancements with Artificial Intelligence:
4) Application of NLP Algorithms:
Latent Dirichlet Allocation: Topic modeling algorithm for detecting abstract themes from a collection of documents.
Support Vector Machine: Classification algorithm for detection of underlying human sentiment.
Long Short-Term Memory Network: Type of recurrent neural networks for machine translation used in Google Translate.



---
