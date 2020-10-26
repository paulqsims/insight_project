# Readme

![Test](/app/logo2.jpeg)

## Description

Skincare products can be quite expensive and it is both *time consuming* and *difficult* to find cheaper alternatives ('dupes'), which often requires understanding formulation details and domain knowlege of ingredient importance to find functionally similar products.

[**DupeMySkincare**](https://share.streamlit.io/paulqsims/insight_project/master/app%2Finsight_app.py) is a python web app that simplifies this process for consumers by providing alternative product recommendations using content based filtering. Users can sort results by similarity and price to best balance their interests and can purchase the alternative product by clicking the link provided.

## How to use it

You can try [DupeMySkincare here](https://share.streamlit.io/paulqsims/insight_project/master/app%2Finsight_app.py)!

In the following order, select the product type, brand, and product that you want to 'dupe' and then click the button 'find my dupe' to get a product recommendation.

## How it works

DupeMySkincare uses skincare product details such as the product type (face wash, toner, etc.) and ingredient features to find similar products. Due to the large number of product features and sparse nature of the data, Truncated Singular Value Decomposition (TSVD) is used to reduce the dimensionality before calculating cosine similarities of feature vectors between all products and the input product of interest. The top most similar product is then recommended to the user.