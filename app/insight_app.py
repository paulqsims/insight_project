# Streamlit app

# Import modules

from pandas._libs.tslibs import conversion
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
from pathlib import Path
import streamlit as st

# Custom functions
# def read_markdown_file(markdown_file):
#     return Path(markdown_file).read_text()

# Set root path for project
path = rootpath.detect()

# Read in data
df = (pd.read_csv(f"{path}/data/data_clean.csv",
                  index_col=False))

df_ex = (pd.read_csv(f"{path}/data/app_ex.csv",
                  index_col=0))

# model_linear = LinearRegression()

# # Impute missing values
# df.fillna(df.mean(), inplace=True)

# # divide df into features matrix and target vector
# features = df[['good_ingred_cat_rat_n', 'Ingredient_n']]     #df.iloc[:, :-1]  #all except quality
# target = df['product_rating']

# X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, train_size=0.8,test_size=0.2, random_state=1)

# # fit a model
# lm = linear_model.LinearRegression()
# model = lm.fit(X_train, y_train)
# predictions = lm.predict(X_test)

# Run app
st.title('SkincareDupe')
'A web app to recommond skincare products based on ingredients'

# st.number_input(label = 'Number of Ingredients')

# st.button('Predict Rating')

# num_ingredients = st.number_input("Number of Ingredients", format="%2d", key="ingredient_input")
# # define input
# new_input = [[num_ingredients, 10]] #num_ingredients
# # get prediction for new input
# new_output = model.predict(new_input).round(decimals = 0)
# st.success('The predicted rating is {}'.format(new_output)) 

# User inputs the number of ingredients in a product
#number_ingredients = st.number_input("Number of Ingredients", format="%2d")

# User uploads ingredients
ingredient_input = st.text_input("Ingredient list")

# ingredient_input = "Water, Dimethicone, Aluminum Starch Octenylsuccinate, Dimethicone Crosspolymer, Ammonium Acryloyldimethyltaurate/VP Copolymer, Trisiloxane, Nylon-12, C12-15 Alkyl Benzoate, Ascorbyl Glucoside, Glycerin, Caprylyl Glycol, Polyacrylamide, Xanthan Gum, Fragrance, C13-14 Isoparaffin, Sodium Hyaluronate, Sodium Lactate, Hydrolyzed Myrtus Communis Leaf Extract, Sodium Hydroxide, BHT, Disodium EDTA, Polysorbate 20, Laureth-7, Retinol, Sodium PCA, Sorbitol, Proline, Hinokitiol, Mica, Titanium Dioxide"

# ingredient_input = ingredient_input.split(',')
# ingredient_input = [ingredient.strip(' ').lower() for ingredient in ingredient_input]



# Use ingredient input to predict rating from model
# if number_ingredients:
#      # define input
#      # num_ingredients = st.number_input("Number of Ingredients", format="%2d", key="rating_output")
#      new_input = [[number_ingredients, 10]] #num_ingredients
#      # get prediction for new input
#      new_output = model.predict(new_input).round(decimals = 0)
#      new_output = str(new_output).strip('[.]') # convert to text
#      st.success(f'The predicted rating is {new_output} stars') 
#      rating_text = read_markdown_file(f"{path}/app/rating_text.md")
#      st.markdown(rating_text, unsafe_allow_html=True)

df_ex2 = df_ex.drop(['predicted_cluster_label'], axis=1)
df_ex2 = df_ex2.rename(columns={'predicted_cluster_prob':'Similarity'})
df_ex2['Similarity'] = round(df_ex2['Similarity'], 2)*100

# Convert ingredients to text
if ingredient_input:
     # define input
    #ingredients = st.text_input("Number of Ingredients")
    #print(ingredients)
    # new_input = [[pd.Series(ingredients) for ingredient in str(product).split(',')] for product in df.Ingredients] #num_ingredients
    #  # get prediction for new input
    #  new_output = model.predict(new_input).round(decimals = 0)
    #  new_output = str(new_output).strip('[.]') # convert to text
    st.success(f'The top most similar products are:') 
    st.table(df_ex2.assign(hack='').set_index('hack'))
    #  rating_text = read_markdown_file(f"{path}/app/rating_text.md")
    #  st.markdown(rating_text, unsafe_allow_html=True)

