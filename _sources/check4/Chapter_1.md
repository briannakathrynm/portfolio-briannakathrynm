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

#  Submission 4 - Chapter 1
I've demonstrated **workflow level 3**, **clustering level 3**, **evaluate level 3**, **compare level 2 and 3**, and **optimize level 3** in this chapter. 


**Workflow Level 3**
<br>
Scope, choose an appropriate tool pipeline and solve data science problems, describe strengths and weakensses of common tools.


**Clustering Level 3**
<br>
Apply multiple clustering techniques, and interpret results


**Evaluate Level 3**
<br>
Evaluate a model with multiple metrics and cross validation.
<br><br>
This skill is present in the multiple metrics used to assess the different types of clustering methods, as well as performing cross-validation in the Iris dataset.


**Compare Level 2**
<br>
Compare model classes in specific terms and fit models in terms of traditional model performance metrics.
<br><br>
**Compare Level 3**
<br>
Evaluate tradeoffs between different model comparison types.


**Optimize Level 3**
<br>
Select optimal parameters based of mutiple quantitative criteria and automate parameter tuning.


## Workflow Level 3


For this section, I will go over my process for the next section (which involves clustering and other skills). I will also go over what I want to achieve using these models, and other tools that could also be used to replace what I am using.

**Common Tools in Data Science**
<br>
There are many different tools that can be used and applied in data science (whether for machine learning purposes, or just statistical). I will list some of these tools that I have personally used before below:
* SKLearn
* SciPy
* Matplotlib
* Seaborn

**Why did I choose the tools I did for this assignment?**
<br><br>
**SKLearn vs. SciPy**
<br>
For this assignment I chose to use a vast majority of methods/classifers from **SKLearn**. In this class, I primarily use SKlearn, but I have used Scipy before. SKLearn is definitely easier, as its documentation provides fully thoguht-out examples and very clear source code of their functions/methods. Scipy is usually more devoted to deeper kinds of calculations (I've used it previously for fourier transformations and signal processing). However, because this class primarily uses SKLearn, I chose to also use that for the majority of my portfolio.

**Strengths of SKLearn:**
* Main strengths are accessiblity, adaptability, and simplicity.
* Very easy for beginners to grasp basic concepts.
* License allows you to upstream changes vwithout restrictions on commerical use.

**Strengths of SciPy**
* Open-source, does not cost anything
* Available libraries make it easy to convert to C or Fortran code
* Has a lot of correlating libraries with Numpy which makes it easy to use the two together


**Seaborn vs. Matplotlib**
<br>
For this assignment, I used both matplotlib and seaborn. I usually like to add more customizations using matplotlib (such as titles, axis rotation, etc) but I prefer the different colore palettes of seaborn. When choosing a tool, you have to keep in mind what is best for visualizing your data. Both seaborn and matplotlib provide an excellent amount of different visualizations, but each different library can provide better visualizations than others. For example, I prefer using **matplotlib's scatterplot function** over seaborn's because of the usability and aesthetics.

**Strengths of Seaborn**
* Use default themes that are aesthetically pleasing.
* You can set custom color palettes.
* Makes attractive statistical plots.
* Easily and flexible for displaying distributions.

**Strengths of Matplotlib**
* Can be automated to adapt to the data that it receives as input. 
* Simple and easy to grasp for beginners.
* asier to use for people who have had prior experience with Matlab.


Now, I will talk about my process for the **Clustering** section of this portfolio (which is the next section). 

For this section, I wanted to predict the quality of wine (high/low) depending on different predictors. To do this, I had to:

* Encode the color of the wine to make it easier to manipulate, along with other categorical variables.
* Define what a 'high' anmd 'low' quality wine was, without being too biased.
* Determine what type of clustering I wanted to use, and how it performed.
* Determine if there was a better way to come to this prediction (in terms of other models, or different types of algorithms). 

I enjoy making a list of what I want to accomplish before I start working on a problem, so this is why this section comes first in this portfolio.

I explain more about why I chose my clustering methods in the **Compare Section** of this portfolio, which I believe should also count for **Workflow :)**.


## Clustering Level 3


I was only missing the descriptions of the clustering techniques to earn this skill in my last check, so I will add descriptions as necessary.

```python
# Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score
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
# Exploring classifications: by quality
print(wine_data['quality'].unique())
```

```python
# Encoding color as 0=white, 1=red
wine_data["color"] = wine_data.color.apply(lambda x: 1 if x == "red" else 0)
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

**Method 1: Using KMeans**
<br><br>
*What is K-Means Clustering?*
<br>
* Partitions datasets into K defined subgroups (non-overlapping) called clusters
* Each data point can only belong to one group
* K-Means then assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid is at the minimum


*How does K-Means Clustering Work?*
* Specify the number of clusters (K)
* Intialize centroids by selecting K points for the centroids
* Iterate until there is no change in the centroids, i.e. the clusters are not changing anymore


**Source:** 
* https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

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
<br><br>
*What is Mean Shift Clustering?*
<br>
* Mean Shift is formed on the idea of KDE (kernel density estimation) which is a method used to estimate the underlying distribution of the data.
* A kernel/weighting function is placed on each point in the dataset, and then updates the centroids of the dataset based on the processing of the data.
* Aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. 


*How does Mean Shift Clustering Work?*
*  At every iteration the kernel is shifted to the centroid or the mean of the points within it.
* The method of calculating this mean depends on the choice of the kernel. In this case if a Gaussian kernel is chosen instead of a flat kernel, then every point will first be assigned a weight which will decay exponentially as the distance from the kernel's center increases.
* At convergence, there will be no direction at which a shift can accommodate more points inside the kernel, and iteration most likely will end.


**Sources:** 
<br>
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
* https://www.sciencedirect.com/science/article/pii/S0047259X14002644?via%3Dihub

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(X)
msc = MeanShift(bandwidth=bandwidth, bin_seeding=True)
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
<br><br>
*What is Agglomerative Clustering?*
<br>
* Partitions datasets into K defined subgroups (non-overlapping) called clusters
* One of the most common types of hierarchial clustering
* Agglomerative clustering works in a “bottom-up” manner. (objects are considered leafs initially, and then are combined into nodes (clusters).


*How does Agglomerative Clustering Work?*
* Starts by treating each object as a singleton cluster.
* Pairs of clusters are successively merged until all clusters have been merged into one big cluster containing all objects.
* The result is a tree-based representation of the objects, named dendrogram.


**Sources**:
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
* https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering/

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

Now, let's evaluate using **cross validation~** as well as **accuracy scores**, Which I always forget to do, but I am definitely remembering to do this time.

```python
# Using a loop to change test sizes
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
testsize_range = [n/10 for n in range(1,7)]
testsize_scores_dt = []
crossval_scores_dt = []
dt = DecisionTreeClassifier()
for testsize in testsize_range:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testsize, random_state=0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    # See, I remembered this time
    cross_val = cross_val_score(dt, X, y)
    scores = (accuracy_score(y_test, y_pred))
    testsize_scores_dt.append(scores.mean())
    crossval_scores_dt.append(cross_val.mean())
```

```python
final_results = pd.DataFrame({'test_size': testsize_range,
                           'score': testsize_scores_dt, "cross validation": crossval_scores_dt})
final_results
```

As we can see, it seems that a **test_size of 0.3 seems to work best for the Iris dataset**. Accurracys of 1 might mean that the dataset is too overfit, or it might be overtrained. The **cross validation score** for test_size 0.3 also fits within the accepted range, so this test size would work perfectly.


Just to make sure I absolutely get **Level 3 for Evaluate**, I will perform KNN on this same dataset and then also perform cross-validation 15 times.

```python
# Using similar loop from before
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(1,40,2))
crossval_scores_knn = []
acc_scores_knn = []
# Using 15 fold cross validation
for k in k_range:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
    # See, I remembered this time
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    # Cross Val score
    cross_val = cross_val_score(knn, X_train, y_train, cv=15)
    crossval_scores_knn.append(cross_val.mean())
    # Accuracy Score
    accuracy_score_knn = knn.score(X_train, y_train)
    acc_scores_knn.append(accuracy_score_knn)
```

```python
# Compiling into dataframe
cv_results = pd.DataFrame({'k': k_range, "cross validation": crossval_scores_knn,
                          "accuracy": acc_scores_knn})
cv_results
```

We can see that **K=7** has the best cross validation score **AND** accuracy (CV Score = 0.961905, Acc = 0.971429).


```python
from sklearn.neighbors import KNeighborsClassifier
# Using our optimal test size from earlier, as well as our optimal k
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
knn2 = KNeighborsClassifier(n_neighbors=7)
knn2.fit(X_train, y_train)
# Predicting values
pred_knn2=knn2.predict(X_test)
print(classification_report(y_test, pred_knn2))
```

```python
# Computing CV again just to show
cross_val = cross_val_score(knn2, X_train, y_train, cv=15)
score_knn_1 = (accuracy_score(y_test, pred_knn2))
print(cross_val)
```

## Compare Level 2


For this section, I will be comparing the two above models (Decision Trees and KNN, as well as Naive Bayes just for fun), as well as the different metrics used to evaluate them, as well as the trade-offs when applying them.


**Comparing Decision Trees, Naive Bayes, and KNN in Specific Terms**
<br>
Both methods are used for classification, but there are many differences in the three models.


**Naive Bayes**
* Supervised learning.
* Linear classifer (unlike KNN).
* Highly accurate when applied to *big data* or large feature sets.
* More hyperparameters than KNN (alpha and beta).
* Does not suffer from the curse of dimensionality.


**Decision Trees**
* Supervised learning.
* Easiest to explain and understand.
* Over-fitting is a major problem with decision trees.
* Works best for a small number of classes.
* Outperformed by KNN and GNB when it comes to *rare occurences* such as outliers, as decision trees sometime require pruning.


**KNN**
* Unsupervised learning.
* All the features must be numeric.
* Doesn't require training.
* You would want to choose KNN over GNB if there is high conditional independence among predictors.
* Suffers from the curse of dimensionality in some cases (more variables/features leads to sample size growing exponentially).


In The past section involving the **Iris Dataset**, I have used both a Decision Tree Classifer, Naive Bayes Classifer, as well as a KNN Classifer to yield results. 
<br><br>
Previously for the DT Classifer and the KNN Classifer, I assessed the two model's results using the *score method* as well as the *cross_val_score* function. Below I will compile a dataframe with the different metrics from both of these models, as well as getting new metrics for KNN and GNB.

```python
# Comparing accuracy/cross_vcal_score of test size for KNN
# Using a loop to change test sizes (same as bework best for a small number of classesfore, but for knn)
testsize_range = [n/10 for n in range(1,7)]
testsize_scores_knn = []
crossval_scores_knn = []
knn = KNeighborsClassifier()
for testsize in testsize_range:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testsize, random_state=0)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # See, I remembered this time
    cross_val = cross_val_score(knn, X, y)
    scores = (accuracy_score(y_test, y_pred))
    testsize_scores_knn.append(scores.mean())
    crossval_scores_knn.append(cross_val.mean())
```

```python
# Comparing accuracy/cross_vcal_score of test size for NB
testsize_range = [n/10 for n in range(1,7)]
testsize_scores_gnb = []
crossval_scores_gnb = []
gnb = GaussianNB()
for testsize in testsize_range:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testsize, random_state=0)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    # See, I remembered this time
    cross_val = cross_val_score(gnb, X, y)
    scores = (accuracy_score(y_test, y_pred))
    testsize_scores_gnb.append(scores.mean())
    crossval_scores_gnb.append(cross_val.mean())
```

```python
final_results_knn = pd.DataFrame({'test_size': testsize_range,
                           'score': testsize_scores_knn, "cross validation": crossval_scores_knn})
final_results_gnb = pd.DataFrame({'test_size': testsize_range,
                           'score': testsize_scores_gnb, "cross validation": crossval_scores_gnb})
```

```python
results = pd.merge(left = final_results, right = final_results_knn, on = ["test_size"])
results_final = pd.merge(left = results, right = final_results_gnb, on = ["test_size"])
results_final.rename(columns={"score_x":"Score DT", "score_y":"Score KNN", 
                        "cross validation_x":"Cross Val Score DT",
                        "cross validation_y":"Cross Val Score KNN",
                        "score":"Score GNB", 
                        "cross validation":"Cross Val Score GNB"})
```

As we can see, our Decision Tree did suffer through some *mild overfitting*, as discovered in the previous section. However, it had better **cross_val_score** than the GNB model with the range of test sizes that it was used on. In terms of consistent model performance, KNN performed very well in terms of **cross_val_score** and **score/accuracy**. This is examined more in the previous sections as well. Unlike for other datasets that I've used in the past, the Naive Bayes model did not perform as well as anticipated. This is probably due to the fact that the Iris dataset is not that large, and is perfect in terms of small classes (3 different types of Irises) for the KNN classifer.


## Compare Level 3


Now I will evaluate the different tradeoffs that can be made by choosing from the models discussed in the previous *Compare* section (Decision Trees, Naive Bayes, KNN).


Overall, each of these models has a decision to make in terms of the **bias variance trade-ff** This tradeoff is simply explained as:
* The model is too simple is often reflected in a biased model with fewer features and more regularization.
* The model is too complex when small changes are made that affect the data tremendously, due to high variance with more features and less regularization.

So what do you choose?
* More complex with less bias and more variance?
* More simple with more bias and less variance?


**Trade-Offs of KNN**
<br>
* There is a definite **bias-variance trade-off** when it comes to KNN classification. This means that if bias is reduced, the variance is increased and vice versa.
<br><br>
* *Bias* is caused by highly correlated predictors, or misinterpretting the relationships between the features and the targets in a dataset.
<br><br>
* *Variance* is caused by fluctuations in the training set. High variance can lead to the model having random noise in the data, and leading thje model to pay a lot of attention to the training data and in turn offer fewer generalizations on data that it has not yet seen.
<br><br>
* Together, bias and variance can lead to the model becoming **over or under - fitted**. How can we solve this?


**Solution for KNN**:
* **Cross Validation**, as we have used previously in this Chapter, can introduce a **validation data set** along with the training and testing sets a model usually has.
* Considering what is important in your model can also help in discovering what is more important in terms of bias and variance. Ideally, low bias and low variance would make for a very well-fitted model, but that can not always occur. This is why introducing a validation set is a good solution.


**Trade-Offs of Decision Trees**
<br>
Similarly to KNN, Decision Trees also face a **bias variance trade-off**. However, solving this problem is a little bit different than KNN. **Decision Trees are also known to have *high variance*, as they create specific branches and splits for samples of the training data, that are specific to this data.**
* In a DT, more roughness = more variance while more smoothness = more bias. When a model gets rougher, it gets more complex. When a model gets smoother, it gets less complex.
* To combat this problem, **pruning the tree** is used to compress the data and to reduce the dimensionality of the model, while still keeping complexity.
* Decreasing the accuracy of the model on the training data increases bias. This lowers variance, meaning your model better generalizes to unseen data (as explained in the KNN section).


**Trade-Offs of Naive Bayes**
<br>
Like other models discussed in this section, Naive Bayes also suffers from **bias variance trade-off**. There are also some solutions to this, such as **tuning hyperparameters alpha and beta**, which I will do in the next section for **Level 3 for Optimize**. Here are the different trade-offs for GNB:
* If alpha = 0 (high variance): our model is overfitted
* If alpha is large (high bias): posterior probabilities are the same


**Other Solutions: Trying different Variations of NB**
There are different types of Naive Bayes classifiers that we can choose from, including:
* Bernoulli NB
* Multinomial NB
* Gaussian NB (which we used above)


**Finally, general solutions for solving these trade-offs:**
<br>
With a model with **high bias:**
* Change the optimization algorithm of the model.
* Perform hyper-parameter tuning on the model.
* Switch the type of model.

With a model with **high variance:**
* Perform regularization using pruning (for DT), dropout (KNN/NB), or Lasso/Ridge Regression for Regression Models.
* Get more data to train on
* Try a different model type


## Optimize Level 3


Going off of the previous section and the criteria for the KNN model, I will now optimize the parameters of our model to insure we have the lowest variance and the lowest bias possible.
<br><br>
To do this, I will optimize the following values:
* n_neighbors, as before
* weights
* p (power parameter)

I optimized these parameters briefly when comparing models in the earlier sections, but I want to agaijn try to optimize them even further.

I will be using a **GridSearch** to perform the tuning of these hyperparameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Default is uniform
weights = ["uniform", "distance"]
# Default 2 (euclidean distance)
p = [1, 2]
# Number of neighbors, default 5, as before (same range)
k_range = list(range(1,40,2))

param_grid = {"n_neighbors": k_range,
              "weights": weights,
              "p": p
              }

grid = GridSearchCV(estimator = KNeighborsClassifier(), 
                    param_grid = param_grid, 
                    cv = 15, # 15, as used previously in the above section
                    scoring = 'accuracy', 
                    refit = True)

knn_model = make_pipeline(StandardScaler(), grid)
knn_model.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
```

We can see that with this optimization, the nubmer of neighbors for our KNN classifer is optimized at **3**, our p is optimized at **1 (manhattan_distance)**, and our weights are optimized to be **uniform**.

Now, I will fit a model with these optimized paramters and compare it to the un-optimized version.

```python
# Comparing optimized vs. un-optimized KNN model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
knn3 = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=1)
knn3.fit(X_train, y_train)
# Predicting values
pred_knn3=knn3.predict(X_test)
print(classification_report(y_test, pred_knn3))
```

```python
# Cross-val score and accuracy score for optimized model
# Optimized Model
cross_val_opt = cross_val_score(knn3, X_train, y_train, cv=15).mean()
scores_opt = (accuracy_score(y_test, pred_knn3)).mean()
# Un-optimized Model
cross_val = cross_val.mean()
score_knn_1 = score_knn_1.mean()
```

```python
# Comparing the two models
compare_results = pd.DataFrame({'Cross_Val_Score': [cross_val, cross_val_opt],
                           'Accuracy': [score_knn_1, scores_opt]})
compare_results
```

We can see that the two models compare very similarly to each other, but this is also because the other model was optmized for test_size while our newly optimized model using GridSearch did not account for test size. I think it would be better to also add that in as well.


**In Evaluate Level 3/Compare Level 2 sections, I automated the tuning of hyperparameters using loops. For this section I wanted to use GridSearch to showcase other ways you could tune these hyperparameters. To consider the automation of these parameters, please look at the aformentioned sections I listed here :)**
