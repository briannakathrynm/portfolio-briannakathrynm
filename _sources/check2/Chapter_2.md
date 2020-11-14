---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Chapter 2
I've demonstrated **classification level 3**, **evaluate level 2 (and 3, if applicable)** and **clustering level 3** in this chapter.


**Classification**
<br>
Fit and apply classification models and select appropriate classification models for different contexts


**Clustering**
<br>
Apply multiple clustering techniques, and interpret results


**Evaluate**
<br>
Level 2: Apply basic model evaluation metrics to a held out test set
<br>
Level 3: Evaluate a model with multiple metrics and cross validation


In this chapter I will be looking at the same dataset I explored in *Assignment #9* previously. I'll include some of the same EDA and methods as well.


## Classification Level 3

```python
# Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
```

```python
# Loading data
wine_data = pd.read_csv("winequality.csv")
wine_data.head()
```

```python
wine_data.shape
```

```python
# Visualizations of data before
plt.figure(figsize=(10,10))
sns.heatmap(wine_data.corr(),annot=True,linewidth=0.5,center=0,cmap='flare')
plt.show()
```

```python
# More visualization
sns.pairplot(wine_data, vars=['chlorides','density', 'alcohol'], hue='quality', palette="flare")
```

```python
# Exploring classifications: by quality
print(wine_data['quality'].unique())
```

```python
# More visualization of classes with qualtiy and alcohol
sns.FacetGrid(wine_data, hue="quality", height=6, palette="flare").map(plt.scatter, "alcohol", "density").add_legend()

plt.show()
```

```python
# Encoding color as 0=white, 1=red
wine_data.color.apply(lambda x: 1 if x == "red" else 0)
wine_data
```

```python
# Separating Quality into High/Low
quality = wine_data["quality"].values
temp = []
for num in quality:
    if num<5:
        temp.append("Low")
    else:
        temp.append("High")
temp = pd.DataFrame(data=temp, columns=["ranking"])
data = pd.concat([wine_data,temp],axis=1)
data.drop(columns="quality",axis=1,inplace=True)
```

```python
data.head()
```

**First Method: KNN Classification**

```python
# Separating target from feature variables
X= data.iloc[:,:-1].values
y=data.iloc[:,-1].values
```

```python
# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
```

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

```python
pred_knn=knn.predict(X_test)
print(classification_report(y_test, pred_knn))
```

Wow! The accuracy is really high with this classifier. Lets explore other options just in case.


**Second Method: Random Forest**

```python
# Train, test, split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.2,random_state=0)
```

```python
# Building the forest
rfc = RandomForestClassifier()
rfc.fit(X_train2, y_train2)
pred_2 = rfc.predict(X_test2)
print(classification_report(y_test2, pred_2))
```

**Third Method: Naive Bayes**

```python
# Train, test, split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size = 0.2,random_state=0)
```

```python
# Applying NB
gnb = GaussianNB()
gnb.fit(X_train3, y_train3)
y_pred3 = gnb.predict(X_test3)
```

```python
gnb.score(X_test3,y_test3)
```

```python
gnb.predict_proba(X_test3)
```

```python
print(classification_report(y_test3, y_pred3))
```

```python
# Combining Results of Classifiers
results = pd.DataFrame({'model': ["KNN","Random Forest","Naive Bayes"],
                           'accuracies': [accuracy_score(y_test,pred_knn),accuracy_score(y_test2,pred_2),accuracy_score(y_test3,y_pred3)]})
results
```

Based on the table above, the best classifier for the **Wine Quality** Dataset would be the **Random Forest** classifier, which has a very tiny difference between the **KNN** classifier.


## Clustering Level 3


**Method 1: Using KMeans**
<br><br>
I previously used KMeans for clustering in *Assignment #9*, so I'll briefly touch upon that now and then explore other methods.

```python
# Separating target from feature variables
X= data.drop(["ranking"], axis = 1)
y=data["ranking"]
```

```python
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
cent = kmeans.cluster_centers_
```

```python
y_kmeans = kmeans.fit_predict(X)
y_kmeans
```

```python
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(X.columns.values))
```

```python
# Plotting relationship between alcohol and density when predicting quality of wine
fig, ax = plt.subplots(1, 1)
X.plot.scatter(x='alcohol', y='density', c= y_kmeans, figsize=(10,8), colormap='flare', ax=ax, mark_right=False)
centroids_df.plot.scatter(x='alcohol', y='density', ax = ax, s = 80, mark_right=False)
```

```python
# Compiling different metrics
metrics1 = pd.DataFrame({'metric': ["Silhouette","Rand Score","Mutual Info"],
                           'score': [silhouette_score(X, y_kmeans),adjusted_rand_score(y, y_kmeans),adjusted_mutual_info_score(y,y_kmeans)]})
metrics1
```

**Method 2: Using Mean Shift**

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(X)
msc = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
```

```python
msc.fit(X)
```

```python
labels = msc.labels_
cluster_centers = msc.cluster_centers_
```

```python
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)
```

```python
y_msc = msc.predict(X)
y_msc
```

```python
centroids_df = pd.DataFrame(cluster_centers, columns = list(X.columns.values))
centroids_df.head()
```

```python
# Plotting relationship between alcohol and density when predicting quality of wine
fig, ax = plt.subplots(1, 1)
X.plot.scatter(x='alcohol', y='density', c= y_msc, figsize=(10,8), colormap='flare', ax=ax, mark_right=False)
centroids_df.plot.scatter(x='alcohol', y='density', ax = ax, s = 80, mark_right=False)
```

```python
# Compiling different metrics
metrics2 = pd.DataFrame({'metric': ["Silhouette","Rand Score","Mutual Info"],
                           'score': [silhouette_score(X, y_msc),adjusted_rand_score(y, y_msc),adjusted_mutual_info_score(y,y_msc)]})
metrics2
```

**Method 3: Agglomerative Clustering**

```python
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5)
ac.fit(X)
```

```python
ac.labels_
```

```python
y_ac = ac.fit_predict(X)
```

```python
# Plotting relationship between alcohol and density when predicting quality of wine
fig, ax = plt.subplots(1, 1)
X.plot.scatter(x='alcohol', y='density', c= y_ac, figsize=(10,8), colormap='flare', ax=ax, mark_right=False)
```

```python
# Compiling different metrics
metrics3 = pd.DataFrame({'metric': ["Silhouette","Rand Score","Mutual Info"],
                           'score': [silhouette_score(X, y_ac),adjusted_rand_score(y, y_ac),adjusted_mutual_info_score(y,y_ac)]})
metrics3
```

Now, I will compile all of the different accuracies, along with the scoring method used for all of the different kinds of clustering.

```python
# Combining all dataframes
result1 = pd.merge(left = metrics1, right = metrics2, how='inner', on=["metric"])
result = pd.merge(left = result1, right = metrics3, how='inner', on=["metric"])
result.rename(columns={"score_x":"K-means", "score_y":"Mean Shift", "score":"Agglomerative Clustering"})
```

From the above table, we can see that **K-Means Clustering** had an overall better silhouette score, **Mean Shift** had a better **Mutual Info** score, and **Agg. Clustering** had a better **Rand Score**. 


## Evaluate Level 2/3


You can see from the multiple metrics I used throughout this portion (both in the clustering and classification sections) that this warrants **evaluate level 2** for applying basic model evaluation metrics, and *part* of **level 3** for using multiple metrics with multiple different types of classifiers.


## Evaluate Level 3


I will now perform some cross-validation for the **Iris Dataset** shown in class, along with using different train and test sizes for the data.

```python
# Loading data
iris_df = pd.read_csv('Iris.csv')
```

```python
# Brief visualization
sns.pairplot(iris_df, hue='Species', palette="flare")
```

```python
# Target and feature variables
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = iris_df['Species'].values
```

```python
# Using a loop to change test sizes, correction from previous assignments!
from sklearn.metrics import accuracy_score
testsize_range = [n/10 for n in range(1,7)]
testsize_scores = []
dt = DecisionTreeClassifier()
for testsize in testsize_range:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testsize, random_state=0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    scores = (accuracy_score(y_test, y_pred))
    testsize_scores.append(scores.mean())
```

```python
final_results = pd.DataFrame({'test_size': ["0.1","0.2","0.3", "0.4", "0.5", "0.6"],
                           'score': testsize_scores})
final_results
```

As we can see, it seems that a **test_size of 0.3 seems to work best for the Iris dataset**. Accurracys of 1 might mean that the dataset is too overfit, or it might be overtrained.
