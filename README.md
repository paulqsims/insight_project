# Readme

![Test](/app/logo2.jpeg)

## Description

Skincare products can be quite expensive and it is both *time consuming* and *difficult* to find cheaper alternatives ('dupes'), which often requires understanding formulation details and domain knowlege of ingredient importance to find functionally similar products.

[**DupeMySkincare**](https://share.streamlit.io/paulqsims/insight_project/master/app%2Finsight_app.py) is a python web app that simplifies this process for consumers by providing alternative product recommendations using content based filtering. Users can sort results by similarity and price to best balance their interests and can purchase the alternative product by clicking the link provided.

## How to use it

You can try [DupeMySkincare here](https://share.streamlit.io/paulqsims/insight_project/master/app%2Finsight_app.py)!

In the following order, select the product type, brand, and product that you want to 'dupe' and then click the button 'find my dupe' to get a product recommendation.

## How it works

DupeMySkincare uses skincare product details such as the product type (face wash, toner, etc.) and ingredients to find similar products, resulting in a recommendation of the most similar product. 

In technical terms, [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) converts the product text information into features, reduces feature dimensionality, and finds similarities between products based on these features, with the most similar product being recommended to the user depending on their recommendation priorties (lowest price, most similar, etc.). 