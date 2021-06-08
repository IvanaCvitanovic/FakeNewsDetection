import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


dataset = pd.read_csv('dataset_final.csv')
print(dataset.shape)
#separating words from articles labeled as true and fake
true = dataset[dataset.label == 0]
fake = dataset[dataset.label == 1]


stop_words = set(stopwords.words('english')) 
tokenized_true = []
tokenized_fake = []

#tokenizing datasets
tokenized_true = true['text'].apply(word_tokenize)
tokenized_fake = fake['text'].apply(word_tokenize)


filtered_true = []
filtered_fake = []

for example in tokenized_true:
    for w in example:
        filtered_true.append(w.lower())
table = str.maketrans('', '', string.punctuation)
filtered_true = [w.translate(table) for w in filtered_true]
# remove remaining tokens that are not alphabetic
filtered_true = [word for word in filtered_true if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_true = [w for w in filtered_true if not w in stop_words]

    
for example in tokenized_fake:
    for w in example:
        filtered_fake.append(w.lower())
table = str.maketrans('', '', string.punctuation)
filtered_fake = [w.translate(table) for w in filtered_fake]
# remove remaining tokens that are not alphabetic
filtered_fake = [word for word in filtered_fake if word.isalpha()]

filtered_fake = [w for w in filtered_fake if not w in stop_words]
            
  

print('true')
print(len(filtered_true))
print('fake')
print(len(filtered_fake))



from collections import Counter


counter_true = Counter(filtered_true)
counter_fake = Counter(filtered_fake)

true_dict = dict(counter_true)
fake_dict = dict(counter_fake)

gamma_measure_dict = {}

for word in true_dict:
    if word in fake_dict:
        fake_freq = fake_dict.get(word)
    else:
        fake_freq = 0
    true_freq = true_dict.get(word)
    gamma_measure_dict[word] = (fake_freq - true_freq) / (fake_freq + true_freq)
    
for word in fake_dict:
    if word in gamma_measure_dict:
        continue
    else:
        fake_freq = fake_dict.get(word)
        if word in true_dict:
            true_freq = true_dict.get(word)
        else:
            true_freq = 0
        gamma_measure_dict[word] = (fake_freq - true_freq) / (fake_freq + true_freq)

    
fake_tokens = []
true_tokens = []
neutral_tokens = []

for word in gamma_measure_dict:
    if word in fake_dict:
        fake_freq = fake_dict.get(word)
    else:
        fake_freq = 0
    if word in true_dict:
        true_freq = true_dict.get(word)
    else:
        true_freq = 0
    
    #if fake_freq - true_freq != 0:
    if gamma_measure_dict.get(word) > 0.7 and fake_freq > 5:
        fake_tokens.append(word)
    if gamma_measure_dict.get(word) < -0.7 and true_freq > 5:
        true_tokens.append(word)
    if abs(gamma_measure_dict.get(word)) < 0.1 and abs(fake_freq - true_freq) < 5:
        neutral_tokens.append(word)

print(len(fake_tokens))
print(len(true_tokens))
print(len(neutral_tokens))