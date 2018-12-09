# Genre Classification based on Plot

## How to run
First you must install all libraries, simply do:
```
$ pip install -r requirements.txt
```
After that you must run the classifier:
```
$ python classifier.py
```

This will generate two files, "classifier.joblib" and "vectorizer.joblib", now you don't need to run the classifier again you can just run the "main.py".

## The idea
Streaming or movie selling websites really use recommendations in order to increase their views/sells. Genre is a really good comparison point between two movies and manually tagging the genre is a really tedious process that I attempt to eliminate by having an automated tagger. This can be really useful to have lists of genres.

### Implementation
This project is a single label classification problem where given a plot we try to guess what the genre of that movie is. Multiple algorithms were tested in order to single out the best for this problem. 
#### Data pre-process
In order to process the text from the plots I used the TFID vectorizer provided in the sckit library, removing all english stopwords as well as a WordNetLemmatizer from NLTK.
#### Features Selected
* Movie Plot
* Movie Genre

The data was obtained from IMDB using a web-scrapper from this [repo](https://github.com/ishmeetkohli/imdbGenreClassification)


Numbers
-------
Here are some numbers for the data:
* Number of titles collected : 4456
* Genres Collected : 19


|               |               |       |
| ------------- |:-------------:| -----:|
| Comedy      | Action        | Drama |
| Animation   | Documentary   | Adventure |
| Biography   | Horror        | Fantasy |
| Mystery     | Romance       | Sci-Fi |
| Family      | Thriller      | Short |
| War         | Musical       | Western |
| Crime        |              |       |    

The distribution is very "uneven" as you can see in the following image:
![chart](https://user-images.githubusercontent.com/34111347/49692182-26b0b000-fb4c-11e8-9f23-8fe661d98338.png)

## Models used
All models were implemented using sckit with the exception of the neural network that was implemented using Tensorflow

### The results:
|               |               |       |
| ------------- |:-------------:| -----:|
|  Model             | Accuracy          |
| SVM      | 49,73%        |
| RandomForestClassifier   | 39.86%   |
| DecisionTreeClassifier   | 30.43%        |
| KNeighborsClassifier     | 35.64%       |
| LogisticRegression      | 48.65%      |
| Neural Network         | 16.43%       |

## Why were the results so bad?
The first is an obvious reason because a lot of movies fit in more than one genre and the classification is usually ambiguous just with a small plot summary. The second one is the small size of the dataset and the small amount of movies from diferent genres being very "comedy" based, a bigger dataset with more diverse genres would have better results. 
