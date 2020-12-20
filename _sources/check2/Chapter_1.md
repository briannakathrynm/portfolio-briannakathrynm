---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

#  Submission 2 - Chapter 1
I've demonstrated level 3 for regression in this chapter.


**Regression Level 3**
<br>
Can fit and explain nonlinear regression


## Regression Level 3
**What is non-linear regression?**
<br>
In the most simple terms, non-linear regression is regression that is **not linear**. However, this also means that non-linear regresssion can be fit to a bunch of different curves because its definition is so fluid. Below I will be fitting a model with both linear and non-linear regresssion.

```python
# Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
```

```python
# Loading data
housing = pd.read_csv("housing.csv")
housing.head()
```

```python
# Exploring more about the data
housing.shape
```

```python
housing.describe()
```

```python
# Looking at correlation between variables
cor = housing.corr()
sns.heatmap(cor)
```

For linear regression analysis, I will be taking a look at the most correlated variable for our *target variable*, which in this case is **median_house_value**.

The msot correlated variable with our target variable seems to be **median_income**.

```python
# Adjusting our data
housing = housing.drop(["households","total_bedrooms", "housing_median_age", "longitude","latitude","total_rooms","population","ocean_proximity"], axis=1)
housing.head()
```

```python
# Splitting our data
X = housing.drop("median_house_value", axis=1)
y = housing["median_house_value"]
```

```python
X
```

```python
y
```

```python
plt.scatter(X, y, cmap = "tab20b", alpha=0.25)
```

By looking at the above scatterplot, we can infer that  there is a **positive linear relationship** with someone with a hgih income and a more expensive house.

```python
# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
```

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

```python
# Predicting values
y_pred = lr.predict(X_test)
y_pred
```

```python
# Evaluating model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

```python
r2
```

```python
rmse
```

```python
lr.coef_
```

From this we can see that:
* Our R2 score falls in the range between 0 and 1
* Our RMSE could be lower, but it is good enough for regression purposes
* Our model has a slope (coeff) of 42291.42

```python
# Visualization of our model
plt.scatter(X_train, y_train, s=10, alpha=0.25)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
# Plotting predictions
plt.plot(X_test, y_pred, color='g')
plt.show()
```

We can see if the data has any non-linear characteristics by visualizing the data with a **residual plot**. According to seaborn documentation, the **residplot()** function can be a useful tool for checking whether the simple regression model is appropriate for a dataset.

```python
# Identifying if the data is non-linear
sns.set_theme(style="whitegrid")
residual = y_test - y_pred
sns.residplot(x=residual, y=y_pred, scatter_kws={'alpha': 0.5}, lowess=True, line_kws={'color': 'green', 'lw': 1, 'alpha': 0.8})
plt.show()
```

The residual line (green) is clearly not straight, which means that a linear model may not be the best fit for our data. In this case, I will try to fit a **polynomial regression model** for our data. Polynomial Regression is defined as a form of regression analysis in which the relationship between the independent variable **x** and the dependent variable **y** are modelled as an nth degree polynomial in **x**.

```python
# Fitting the model
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
```

```python
plt.scatter(X_train, y_train, color="green")
plt.plot(X_train, pol_reg.predict(poly_reg.fit_transform(X_train)))
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.show()
```

```python
# Predicting values
X_poly = poly_reg.fit_transform(X_test)
y_pred_2 = pol_reg.predict(X_poly)
```

```python
# Evaluating the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred_2))
r2 = r2_score(y_test, y_pred_2)
```

```python
r2
```

```python
rmse
```

By using this new model, our R2 score increased from **0.4564966485656323** with the Linear Regression model, to **0.4628233152699195** with the Polynomial Regression model.
