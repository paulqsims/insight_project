# Streamlit app

# Import modules

import streamlit as st
import rootpath
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
import numpy as np

# Set root path for project
path = rootpath.detect()

# Read in data
df = (pd.read_csv(f"{path}/data/data_clean.csv",
                  index_col=False))

model_linear = LinearRegression()

# Impute missing values
df.fillna(df.mean(), inplace=True)

# divide df into features matrix and target vector
features = df[['good_ingred_cat_rat_n', 'Ingredient_n']]     #df.iloc[:, :-1]  #all except quality
target = df['product_rating']

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, train_size=0.8,test_size=0.2, random_state=1)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# Run app
st.title('Skincare Beautypedia Rating Predictor')

# st.number_input(label = 'Number of Ingredients')

# st.button('Predict Rating')

num_ingredients = st.number_input("Number of Ingredients", format="%2d", key="ingredient_input")
# define input
new_input = [[num_ingredients, 10]] #num_ingredients
# get prediction for new input
new_output = model.predict(new_input).round(decimals = 0)
st.success('The predicted rating is {}'.format(new_output)) 

if st.number_input("Number of Ingredients", key="rating_output"):
	result = message.title()
	st.success(result)

st.success('Five stars: Superior. These are industry-leading, world-class products that contain an intriguing amount or combination of research-proven ingredients. \n Four stars: Excellent. These are some of the best products around but have minor formulary, aesthetic, or performance issues that can affect a decision to purchase or results you’ll see. Three stars: Average. May have certain issues such as a lack of key ingredients or a more basic or one-note formula, but might still be worth considering, especially if the price is low. Two stars: Below average. These products are disappointing on many levels, from formula to packaging or value for the money. Not worth strong consideration. One star: Poor. An irritant-laden or otherwise disappointing formula, bad performance, poor packaging, or a combination of these make such products a must to avoid.')