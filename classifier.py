import pandas as pd
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


#Extract data
df = pd.read_csv('trainingSet.csv')

# Graph the dataset
counts = df.Genre1.value_counts()
counts.plot(kind='bar')
plt.show()

#Split data into test and train
x = df['Plot']
y = df['Genre1']

# Tokenization and lemetization
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

# Factorize the labels to numbers
y, names = pd.factorize(y)

# Split train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

#Create and fict a vectorizer to our train data
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

#Choose whichever classifier you prefer by commenting svm and uncommenting your prefered one
# mnb = RandomForestClassifier()
# mnb = tree.DecisionTreeClassifier()
# mnb = KNeighborsClassifier()


# mnb = linear_model.LogisticRegression(n_jobs=1, C=1e5) #Close score
mnb = svm.LinearSVC() #Best score


#Fit the Classifier
mnb.fit(x_train_vectorized, y_train)
# Print the model accuracy
print(mnb.score(x_test_vectorized, y_test))
dump(mnb, 'classifier.joblib')
dump(vectorizer, 'vectorizer.joblib')
