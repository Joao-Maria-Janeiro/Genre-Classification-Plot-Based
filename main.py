from joblib import dump, load
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer


names = ['Animation', 'Action', 'Comedy', 'Adventure', 'Biography', 'Drama',
       'Crime', 'Fantasy', 'Mystery', 'Romance', 'Sci-Fi', 'Documentary',
       'Family', 'Horror', 'Thriller', 'Short', 'Western', 'War', 'Musical']

plot = input('Write the movie plot: ')
mnb = load('classifier.joblib')
vectorizer = load('vectorizer.joblib')
lemmattizer = WordNetLemmatizer()

lem = nltk.word_tokenize(plot)
middle_list = []
for i in range(len(lem)):
    temp = lemmattizer.lemmatize(lem[i])
    middle_list.append(temp)

plot = lemmattizer.lemmatize(' '.join(middle_list))

test_plot = pd.Series(plot)
test_plot = vectorizer.transform(test_plot)
print('Genre: ' + names[mnb.predict(test_plot)[0]])
