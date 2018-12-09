import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

#Extract data
df = pd.read_csv('trainingSet.csv')

#Split data into test and train
x = df['Plot']
y = df['Genre1']

lemmattizer = WordNetLemmatizer()
final_list = []
for plot in x:
    lem = nltk.word_tokenize(plot)
    middle_list = []
    for i in range(len(lem)):
        temp = lemmattizer.lemmatize(lem[i])
        middle_list.append(temp)
    final_list.append(middle_list)

i = 0
for list in final_list:
    x[i] = ' '.join(list)
    i+=1
x = pd.Series(x)

y, names = pd.factorize(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

#Create and fict a vectorizer to our train data
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

x_train_vectorized = x_train_vectorized.toarray()
x_test_vectorized = x_test_vectorized.toarray()


# Preprocess the data by setting all array to the same size using padding
train_data = keras.preprocessing.sequence.pad_sequences(x_train_vectorized,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=2000)

test_data = keras.preprocessing.sequence.pad_sequences(x_test_vectorized,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=2000)


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.models.Sequential([
    keras.layers.Dense(20, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(2000,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Model features
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create a cross validation set
x_val = train_data[:1000]
partial_x_train = train_data[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]



# Train the model for 40 epochs in mini-batches of 512 samples. This is 40 iterations over all samples in the x_train and y_train tensors. While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=2)

# Print the model accuracy
results = model.evaluate(test_data, y_test)
print(results)
