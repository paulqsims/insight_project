# Streamlit app

# Import modules

from pandas._libs.tslibs import conversion
import streamlit as st
import rootpath 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import heapq as hq

# Custom functions
# def read_markdown_file(markdown_file):
#     return Path(markdown_file).read_text()

# Set root path for project
path = rootpath.detect()

# Read in data
# Use list comprehension to read in all files
df = pd.read_csv(f"{path}/data/data_clean.csv", index_col=0).reset_index(drop=True)

#df = pd.read_csv("data_clean.csv", index_col=0).reset_index(drop=True)


#df_ex = (pd.read_csv(f"{path}/data/app_ex.csv",
#                  index_col=0))

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
st.title('DupeMySkincare')
'A web app to recommond skincare productss based on ingredients'

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

features = df.copy().drop(['product_type','active', 'brand', 'price','size','ratings', 'total_reviews','link','price_oz'],axis=1).set_index('product')

# Add a blank row for default selection
#df.append(pd.Series(), ignore_index=True)

# input_choice = st.selectbox("Select a brand",
#                           ('Paste ingredients', 'Search our database'),key='a')

prod_type = st.selectbox("1. Select product type", (df['product_type'].unique()),key='a')
if prod_type != 0:
     df2 = df[df.product_type==(f'{prod_type}')]
     brand = st.selectbox("2. Select the product's brand", (df2['brand'].unique()),key='a')
     if brand:
          df3 = df2[df2.brand==(f'{brand}')]
          product = st.selectbox("3. Select the product", (df3['product'].unique()),key='b')

if st.button('Find my dupe!'):
     # st.subheader('Choose the brand and product you want to dupe')
     # st.markdown('Select or type the product type, brand, and product name')
     # prod_type = st.selectbox("Product type", (df['product_type'].unique()),key='a')
     # df2 = df[df.product_type==(f'{prod_type}')]
     # if prod_type != 0:
     #      brand = st.selectbox("Brand", (df2['brand'].unique()),key='a')
     #      df3 = df2[df2.brand==(f'{brand}')]
     #      product = st.selectbox("Product", (df3['product'].unique()),key='b')
     #      if product !=0:
     #           if st.selectbox:
     #product_input=df.loc[df['product']==f'{product}']
     # Calculate cosine similarity for a given product
     res_cosine = cosine_similarity(features.loc[f'{product}',:].to_frame().transpose(), features) 
     res_cosine = res_cosine.reshape(-1)
     res_cosine = pd.DataFrame(res_cosine)
     res_sim=df[['product','brand','product_type','price','size','ratings',
               'total_reviews','link','price_oz']].copy()
     res_sim['similarity']=res_cosine[[0]]
# Round similarity metric
#res_sim['similarity']=round(res_sim['similarity'],2)
# Maybe don't round so you don't have to deal with ties?
#indexNames = res_sim[res_sim['product']=='Essential-C Cleanser'].index
#res_sim.drop(indexNames, inplace=True)
# Sort from top similarity metrics and ignoring self
     top_sim = res_sim.nlargest(5, 'similarity')[1:6]
     #best_sim_score = np.min(max(top_sim['similarity'],min(top_sim['price_oz'])))
     output_rec = top_sim.iloc[0].to_frame().transpose()[['product_type', 'brand','product','similarity', 'price','price_oz','size','link']]
     output_rec['similarity']=output_rec['similarity'].astype(float)
     output_rec['similarity']=output_rec['similarity']
     # keep df of product selected
     tempdf = df.loc[df['product']==f'{product}']
     #total price diff
     price_diff = tempdf['price']-output_rec.iloc[0]['price']
     price_diff = price_diff.astype('float')
     price_diff = price_diff.values[0]
     # price per oz
     price_diff_oz = tempdf['price_oz']-output_rec.iloc[0]['price_oz']
     price_diff_oz = price_diff_oz.astype('float')
     price_diff_oz = price_diff_oz.values[0]
     st.subheader('Try this product instead:')
     st.table(output_rec.style.format({'similarity':'{:.2f}',
     'price':'{:.2f}','price_oz':'{:.2f}','size':'{:.2f}'})) 
     st.markdown(f'Savings (total price difference): ${price_diff:.2f}')
     st.markdown(f'Savings (price per oz): ${price_diff_oz:.2f}')
# else:
#      # User uploads ingredients
#      ingredients = st.text_input("Ingredient list")

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

# df_ex2 = df_ex.drop(['predicted_cluster_label'], axis=1)
# df_ex2 = df_ex2.rename(columns={'predicted_cluster_prob':'Similarity'})
# df_ex2['Similarity'] = round(df_ex2['Similarity'], 2)*100

# Convert ingredients to text
#if ingredient_input:
     # define input
    #ingredients = st.text_input("Number of Ingredients")
    #print(ingredients)
    # new_input = [[pd.Series(ingredients) for ingredient in str(product).split(',')] for product in df.Ingredients] #num_ingredients
    #  # get prediction for new input
    #  new_output = model.predict(new_input).round(decimals = 0)
    #  new_output = str(new_output).strip('[.]') # convert to text
#     st.success(f'The top most similar products are:') 
#     st.table(df_ex2.assign(hack='').set_index('hack'))
    #  rating_text = read_markdown_file(f"{path}/app/rating_text.md")
    #  st.markdown(rating_text, unsafe_allow_html=True)

