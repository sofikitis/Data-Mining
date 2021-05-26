import csv
import numpy as np
import math
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


file = 'onion-or-not.csv'
with open(file, encoding="mbcs") as csvFile:
    my_List = list(csv.reader(csvFile, delimiter=','))

# Remove title from data
my_List.remove(my_List[0])

# Convert list to array
data = np.array(my_List)

# Split label from data
onion_or_not = data[:, -1]


# --- 1 ---
wordArr = []
for row in data:
    sentence = word_tokenize(row[0])
    wordArr.append(sentence)


# --- 2 ---
ps = PorterStemmer()
wordArr = [[ps.stem(word) for word in sentence] for sentence in wordArr]


# --- 3 ---
stop_words = set(stopwords.words('english'))
wordArr = [[word for word in sentence if not word in stop_words] for sentence in wordArr]


# --- 4 ---
frequency_matrix = {}
i = 0
for sentence in wordArr:
    freq_table = {}

    for word in sentence:

        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    frequency_matrix[i] = freq_table
    i = i + 1

tf_matrix = {}
i = 0
for sentence, f_table in frequency_matrix.items():
    tf_table = {}

    count_words_in_sentence = len(f_table)
    for word, count in f_table.items():
        tf_table[word] = count / count_words_in_sentence

    tf_matrix[i] = tf_table
    i = i + 1

word_per_doc_table = {}
for sentence, f_table in frequency_matrix.items():
    for word, count in f_table.items():
        if word in word_per_doc_table:
            word_per_doc_table[word] += 1
        else:
            word_per_doc_table[word] = 1


total_documents = len(wordArr)
idf_matrix = {}
i = 0
for sent, f_table in frequency_matrix.items():
    idf_table = {}

    for word in f_table.keys():
        idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word]))

    idf_matrix[i] = idf_table
    i = i + 1


tf_idf_matrix = {}
i = 0
for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

    tf_idf_table = {}
    for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
        tf_idf_table[word1] = float(value1 * value2)

    tf_idf_matrix[i] = tf_idf_table
    i = i + 1


# --- 5 ---

final = []
max_len = 0
for sentence, f_table in tf_idf_matrix.items():

    arr = []
    c_len = 0
    for word, value in f_table.items():
        arr.append(value)
        c_len = c_len + 1

    if c_len > max_len:
        max_len = c_len
    final.append(arr)

for sen in final:
    for i in range(len(sen), max_len):
        sen.append(0)

features = np.array(final)
label = np.array(onion_or_not)

print(features.shape, label.shape)

# -------- NN --------

data_train, data_test, result_train, result_test = train_test_split(features, label, test_size=0.25)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
clf.fit(data_train, result_train)
pr = clf.predict(data_test)
print(classification_report(result_test, pr, zero_division=0))
