---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Submission 4 - Chapter 2
In this chapter I have demonstrated **unstructured level 3**.


**Unstructured Level 3**
<br>
Apply multiple representations and compare and contrast them for different end results.


## Unstructured Level 3


For this portion of my portfolio, I will be briefly touching upon the different representations of unstructured data covered in **Assignment 12**. 

To refresh, here are the different methods we can use:

**CountVectorizer**
<br>
Returns an encoded vector with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.

**TfidfVectorizer**
<br>
Allows use to tokenize documents, learn similar vocabularies and the frequencies/weights of different works in documents (but not across documents), and allows you to encode new documents.

**HashingVectorizer**
<br>
Uses a one-way hash of words to convert them into integers, and then tokenizes and encodes the documents as needed.


For this portfolio, I will be comparing and contrasting **CountVectorizer** with **TfidVectorizer**. Here are the **similarities and differences** between these two methods:
* TfidVectorizer returns a float instead of an int like CountVectorizer.
* In TfidfVectorizer we consider overall document weightage of a word. It helps us in dealing with most frequent words.
* In CountVectorizer we only count the number of times a word appears in the document which results in biasing in favour of most frequent words.


For this assignment, I will be completing Assignment 12 with both **CountVectorizer** and **TfidVectorizer**, and then comparing the results.

```python
# Imports
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
```

**Using CountVectorizer**

```python
# Loading Data
news = pd.read_csv("fake_or_real_news.csv")
news.head()
```

```python
# Dropping labels that are not necessary
news.drop(labels=['id','title'], axis='columns', inplace=True)
news.head()
# Separate text from labels
text = news['text']
labels = news['label']
# Vectorize Data
counts = CountVectorizer(analyzer = "word")
ng_vec = counts.fit_transform(text).toarray()
# get_feature_names method will return them as a sorted list instead of a dictionary with numbers.
counts_df = pd.DataFrame(ng_vec, columns=counts.get_feature_names())
```

```python
# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(counts_df, labels, test_size = 1000)

# Naive Bayes Model
clf = MultinomialNB()
clf.fit(X_train, y_train)
# Scoring the model
clf.score(X_test, y_test)
# Making predictions on test data
y_pred_test = clf.predict(X_test)
# Making predictions on training data
y_pred_train = clf.predict(X_train)
# Training Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
# Testing Accuracy
test_acc = accuracy_score(y_test,y_pred_test)
```

```python
train_acc
```

```python
test_acc
```

**Using TfidVectorizer**

```python
# Vectorize Data
tfid = TfidfVectorizer(analyzer = "word")
tfid_vec = tfid.fit_transform(text).toarray()
tfid_df = pd.DataFrame(tfid_vec, columns=tfid.get_feature_names())
```

```python
# Train, test, split for TFID
X_train, X_test, y_train, y_test = train_test_split(tfid_df, labels, test_size = 1000)

# Naive Bayes Model
clf2 = MultinomialNB()
clf2.fit(X_train, y_train)
# Scoring the model
clf2.score(X_test, y_test)
# Making predictions on test data
y_pred_test2 = clf2.predict(X_test)
# Making predictions on training data
y_pred_train2 = clf2.predict(X_train)
# Training Accuracy
train_acc2 = accuracy_score(y_train, y_pred_train2)
# Testing Accuracy
test_acc2 = accuracy_score(y_test, y_pred_test2)
```

```python
train_acc2
```

```python
test_acc2
```

Now I will summarize the results of these two vectorizers, and compare/contrast them.

```python
# Compiling dataframe
compare_results = pd.DataFrame({'Train Accuracy': [train_acc, train_acc2],
                           'Test Accuracy': [test_acc, test_acc2]})
compare_results
```

We can see in the compiled dataframe (with 0 being CountVectorizer and 1 being TfidVectorizer), that the accuracies for both the training and testing sets are better with **CountVectorizer**. 

In the future, I would also like to optimize the hyperparamters of these vectorizers, but I don't have enough time. Maybe that will be a fun winter-break project :D
