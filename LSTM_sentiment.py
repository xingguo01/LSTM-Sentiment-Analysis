import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randint

from os import listdir
from os.path import isfile, join
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)       
print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  
print('Negative files finished')
numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))


# Load the vocabulary and the word vectors
wordsList = np.load('wordsList.npy', allow_pickle=True).tolist()
wordVectors = np.load('wordVectors.npy')
wordsList = [word.decode('UTF-8') for word in wordsList]

# Load the ids matrix which contains the indices of the words in the vocabulary
ids = np.load('idsMatrix.npy')
# Parameters
maxSeqLength = 250  # Maximum length of sequence
numDimensions = 300  # Dimensions for each word vector
batchSize = 24
lstmUnits = 64
numClasses = 2
max_features=2000
embed_dim=128

# Define the LSTM model
'''
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=numDimensions, output_dim=numClasses, input_shape=(maxSeqLength,)),
    tf.keras.layers.LSTM(lstmUnits),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(numClasses, activation='softmax')
])
'''
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(maxSeqLength,)))
model.add(tf.keras.layers.Embedding(input_dim=numDimensions, output_dim=numClasses))
model.add(tf.keras.layers.LSTM(lstmUnits))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(numClasses, activation='softmax'))




# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to clean sentences
def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# Functions to generate batches for training and testing
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1]
    return arr, np.array(labels)

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1]
    return arr, np.array(labels)

# Training and evaluating the model
for i in range(10):  # Number of training iterations
    train_data, train_labels = getTrainBatch()
    test_data, test_labels = getTestBatch()
    
    # Convert labels to the correct format for 'sparse_categorical_crossentropy'
    train_labels = np.argmax(train_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    
    model.train_on_batch(train_data, train_labels)
    
    if i % 5 == 0:  # Evaluate every 5 iterations
        loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
        print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        print(train_data,train_labels)
        print(test_data,test_labels)

