---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Submission 1 - Chapter 3
In this submission, I will be attempting to earn **summarize level 3**, **construct level 3**, and **visualize level 3**.


Back in [this chapter](Chapter_1.ipynb), I briefly explored the [simple folk database](simplefolks.sqlite). In this Chapter, I plan on diving deeper and creating more tables, as well as visualizing and summarizing data by those created tables.


### Summarize Level 3

```python
#Imports and data loading
import sqlite3
import pandas as pd
con = sqlite3.connect('simplefolks.sqlite')
cur = con.cursor()
rows = cur.fetchall()
```

This database contains 5 different tables of a simple town. It has information on the town's **people, homes, their pets, politicians, and prison inmates**. Back in Chapter 1, I constructed a couple of tables to show the people and pets of this town, as well as constructing other tables based on simple calculations. In this chapter, I will construct more tables as well as creating new tables based on calculations.


First, lets start with something simple. In this first table, I will find information about the females of this town, and statistics about their age.

```python
# First query
query_1 = pd.read_sql_query("SELECT * FROM people WHERE sex = 'F'", con)
query_1
```

```python
# Calculating summary statistics of first query
query_1.describe()
```

Now, the above calculation is quite simple. It just tells us the various summary statistics about ALL women in the town, with no other factors. Now, I will create a table with different attributes to create a subset of this first table.

```python
# Creating subset of data from first query
subset_1 = pd.read_sql_query("SELECT * FROM people WHERE sex = 'F' AND age >= 35 ORDER BY age DESC", con)
subset_1
```

The above table describes women in the town whose age is above 35 (close to the mean of the data), with their age in decreases order. This once again is a simple table, but next I will dive into some more complex calculations.

```python
# Second query
query_2 = pd.read_sql_query("SELECT * FROM homes", con)
query_2
```

```python
# Calculating summary statistics for above data
query_2.describe()
```

This table shows us the homes in the town, who they are owned by, the area they are located in, and the value of the home. The next subset will show us the **three cheapest** homes in all of the different areas.

```python
# Subset for urban houses
subset_urban = pd.read_sql_query("SELECT * FROM homes WHERE area = 'urban' ORDER BY value ASC LIMIT 3", con)
# Subset for country houses
subset_country = pd.read_sql_query("SELECT * FROM homes WHERE area = 'country' ORDER BY value ASC LIMIT 3", con)
# Subset for suburb houses
subset_suburbs = pd.read_sql_query("SELECT * FROM homes WHERE area = 'suburbs' ORDER BY value ASC LIMIT 3", con)
```

```python
# Calculating summary statistics for urban subset
subset_urban.describe()
```

```python
# Calculating summary statistics for country subset
subset_country.describe()
```

```python
# Calculating summary statistics for suburb subset
subset_suburbs.describe()
```

Based on the summary statistics above, we can see that among the 3 least expensive homes, the minimum value from all the different areas would be in the **country subset** at 42,000. The maximum value from the three least expensive homes at all of the areas would be from the **urban subset** at 190,000. From this data, we can see that the most expensive homes (even from listing the three *least expensive* homes) would be from the **suburb subset** with a minimum of 95,000, and std of 37,527, and a maximum of 160,000.


Now that we are getting more specific, lets list the **3 most expensive homes** that are **not in an urban area** and **not owned by Donald**.

```python
# Third subset
subset_3 = pd.read_sql_query("SELECT * FROM homes WHERE area != 'urban' AND owner_name != 'Donald' ORDER BY value ASC", con)
subset_3
```

### Visualize Level 3
Now, let's take the housing data, and create different visuals and plots based on the price, location, and owner.

```python
# Imports
import seaborn as sns
import matplotlib.pyplot as plt
```

Showing how many houses are owned by each person.

```python
chart_1 = sns.catplot(x="owner_name", kind="count", palette="ch:.25", data=query_2)
chart_1.set_xticklabels(rotation=45)
```

Showing the different values of homes in different areas, note the outliers.

```python
chart_2 = sns.catplot(x="value", y="area", kind="box", palette="ch:.25", data=query_2)
```

Showing different homes owned by people and their corresponding values and areas.

```python
chart_3 = sns.catplot(x="value", y="owner_name", col="area",
                data=query_2, kind="swarm",
                height=4, aspect=.7, palette="ch:.25");
```

### Construct Level 3


In this section I will construct some data tables using data that is not automatically aligned.

```python
# Loading data
friends= pd.read_excel("friends.xlsx")
friends_1 = pd.read_excel("friends_1.xlsx")
```

```python
# Look at first file
friends
```

```python
# Look at second file
friends_1
```

As we can see above, the two datasets are not the same. The first dataset has 7 rows and 5 columns, whereas the second dataset has 5 rows and 4 columns.


The key here is to look for similar column names to merge on, even if the data is not aligned. I will do this below.

```python
# First merge
merge_1 = pd.merge(left = friends, right = friends_1, how='outer', on=['name', 'age', 'sport', 'birthday_month'])
merge_1
```

As we can see, we have some missing NaN values in the additional column that was included in the first dataset **friends**. However, we can fill these missing values using the **mode**, or most common occurence of the other values for **favorite_color**.

```python
friends.favorite_color.mode()
```

As we can see from the code above, the mode for **favorite_color** in the friends dataset is Orange. Now, we can fill the missing values with these occurences.

```python
for column in ['favorite_color']:
    merge_1[column].fillna(merge_1[column].mode()[0], inplace=True)
```

```python
merge_1
```
