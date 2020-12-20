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

# Check 4 Chapter 3
Skills in this chapter:
* Unstructured Level 3
* Workflow Level 3


**Unstructured Level 3**
<br>
Apply multiple representations and compare and contrast them for different end results.


## Unstructured Level 3
For this portion of my portfolio, I will be briefly touching upon the different representations of unstructured data covered in **Assignment 12**. To refresh, here are the different methods we can use:

**CountVectorizer**
<br>
Returns an encoded vector with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.

**TfidfVectorizer**
<br>
Allows use to tokenize documents, learn similar vocabularies and the frequencies/weights of different works in documents (but not across documents), and allows you to encode new documents.

**HashingVectorizer**
<br>
Uses a one-way hash of words to convert them into integers, and then tokenizes and encodes the documents as needed.


For this portfolio, I will be comparing and contrasting **CountVectorizer** with **HashingVectorizer**. Here are the similarities and differences between these two methods:
* HashingVectorizer converts the collection of text documents to a matrix of occurrences, while CountVectorizer converts it to a matrix of counts.
* HashingVectorizer is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory.

CountVectorizer is very straightforward, so I'm mostly highlighting the differences in HashVectorizer.

```python

```
