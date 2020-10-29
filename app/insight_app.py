# Streamlit app

# Import modules

from pandas._libs.tslibs import conversion
import streamlit as st
import rootpath # directory setting
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import heapq as hq # for sorting df by largest values
import base64 # for header image

# Custom functions
# def read_markdown_file(markdown_file):
#     return Path(markdown_file).read_text()

# Set root path for project
path = rootpath.detect()

# Read in data
# Use list comprehension to read in all files
df = pd.read_csv(f"{path}/data/data_clean.csv", index_col=0).reset_index(drop=True)

#df = pd.read_csv("data_clean.csv", index_col=0).reset_index(drop=True)

# Add background image
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(f"{path}/app/logo2.jpeg")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

features = df.copy().drop(['product_type','active', 'brand', 'price','size','ratings', 'total_reviews','link','price_oz'],axis=1).set_index('product')

# Add a blank row for default selection
#df.append(pd.Series(), ignore_index=True)

# input_choice = st.selectbox("Select a brand",
#                           ('Paste ingredients', 'Search our database'),key='a')
st.markdown("")
st.markdown("Please enter details about the product you wish to 'dupe':")

prod_type = st.selectbox("1. Select product type", (df['product_type'].unique()),key='a')
if prod_type != 0:
     df2 = df[df.product_type==(f'{prod_type}')]
     brand = st.selectbox("2. Select the product's brand", (df2['brand'].unique()),key='a')
     if brand:
          df3 = df2[df2.brand==(f'{brand}')]
          product = st.selectbox("3. Select the product you want to 'dupe'", (df3['product'].unique()),key='b')

prod_rec = st.selectbox("4. Prioritize recommendation by:", ('Cheaper total price','Cheaper price per oz', 'Most similar'),key='c')

# def make_clickable(link):
#     # target _blank to open new window
#     # extract clickable text to display for your link
#     text = link.split('=')[1]
#     return f'<a target="_blank" href="{link}">{text}</a>'

def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val,val)

pd.set_option('display.max_colwidth', -1)

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
     if prod_rec == 'Most similar':
        # Sort from top similarity metrics and ignoring self
        top_sim = res_sim.nlargest(5, 'similarity')[1:6]
        #best_sim_score = np.min(max(top_sim['similarity'],min(top_sim['price_oz'])))
        output_rec = top_sim.iloc[0].to_frame().transpose()[['product_type', 'brand','product','similarity', 'price','price_oz','size','link']]
        output_rec['similarity']=output_rec['similarity'].astype(float)
        # Convert to percent
        output_rec['similarity']=output_rec['similarity'] * 100
        # Rename column to percent
        output_rec.rename(columns={'similarity':'similarity (%)',
                                    'size':'size (oz)',
                                'price':'price ($USD)'}, inplace = True)
        # keep df of product selected
        tempdf = df.loc[df['product']==f'{product}']
        #total price diff
        price_diff = tempdf['price']-output_rec.iloc[0]['price ($USD)']
        price_diff = price_diff.astype('float')
        price_diff = price_diff.values[0]
        # price per oz
        price_diff_oz = tempdf['price_oz']-output_rec.iloc[0]['price_oz']
        price_diff_oz = price_diff_oz.astype('float')
        price_diff_oz = price_diff_oz.values[0]
        # Round similarity, price, and price per oz, and size
        # output_rec.round({'similarity (%)':2,'price ($USD)':2,'price_oz':2,
        # 'size (oz)':2})
        output_rec.round(2)
        # Header
        st.subheader('Try this product instead:')
        # Make clickable links
        output_rec= output_rec.to_html(render_links=True)
        # Print product recommendation table with clickable link
        st.write(output_rec, unsafe_allow_html=True)
        # st.table(output_rec.style.format({'similarity (%)':'{:.2f}',
        # 'price ($USD)':'{:.2f}','price_oz':'{:.2f}','size (oz)':'{:.2f}'})) 
        st.markdown('')
        st.markdown(f'Savings (total price difference): ${price_diff:.2f}')
        st.markdown(f'Savings (price per oz): ${price_diff_oz:.2f}')
     elif prod_rec == 'Cheaper total price':
        # Sort from top similarity metrics and ignoring self
        top_sim = res_sim.nlargest(5, 'similarity')[1:6]
        #best_sim_score = np.min(max(top_sim['similarity'],min(top_sim['price_oz'])))
        output_rec = top_sim.nsmallest(1, 'price')
        output_rec['similarity']=output_rec['similarity'].astype(float)
        # Convert to percent
        output_rec['similarity']=output_rec['similarity'] * 100
        # Rename column to percent
        output_rec.rename(columns={'similarity':'similarity (%)',
                                    'size':'size (oz)',
                                'price':'price ($USD)'}, inplace = True)
        # keep df of product selected
        tempdf = df.loc[df['product']==f'{product}']
        #total price diff
        price_diff = tempdf['price']-output_rec.iloc[0]['price ($USD)']
        price_diff = price_diff.astype('float')
        price_diff = price_diff.values[0]
        # price per oz
        price_diff_oz = tempdf['price_oz']-output_rec.iloc[0]['price_oz']
        price_diff_oz = price_diff_oz.astype('float')
        price_diff_oz = price_diff_oz.values[0]
        # Round similarity, price, and price per oz, and size
        # output_rec.round({'similarity (%)':2,'price ($USD)':2,'price_oz':2,
        # 'size (oz)':2})
        output_rec.round(2)
        output_rec = output_rec[['product_type','brand','product','similarity (%)','price ($USD)', 'price_oz','size (oz)','link']]
        # Header
        st.subheader('Try this product instead:')
        # Make clickable links
        output_rec= output_rec.to_html(render_links=True)
        # Print product recommendation table with clickable link
        st.write(output_rec, unsafe_allow_html=True)
        # st.table(output_rec.style.format({'similarity (%)':'{:.2f}',
        # 'price ($USD)':'{:.2f}','price_oz':'{:.2f}','size (oz)':'{:.2f}'})) 
        st.markdown('')
        st.markdown(f'Savings (total price difference): ${price_diff:.2f}')
        st.markdown(f'Savings (price per oz): ${price_diff_oz:.2f}')
     elif prod_rec == 'Cheaper price per oz':
        # Sort from top similarity metrics and ignoring self
        top_sim = res_sim.nlargest(5, 'similarity')[1:6]
        #best_sim_score = np.min(max(top_sim['similarity'],min(top_sim['price_oz'])))
        output_rec = top_sim.nsmallest(1, 'price_oz')
        output_rec['similarity']=output_rec['similarity'].astype(float)
        # Convert to percent
        output_rec['similarity']=output_rec['similarity'] * 100
        # Rename column to percent
        output_rec.rename(columns={'similarity':'similarity (%)',
                                    'size':'size (oz)',
                                'price':'price ($USD)'}, inplace = True)
        # keep df of product selected
        tempdf = df.loc[df['product']==f'{product}']
        #total price diff
        price_diff = tempdf['price']-output_rec.iloc[0]['price ($USD)']
        price_diff = price_diff.astype('float')
        price_diff = price_diff.values[0]
        # price per oz
        price_diff_oz = tempdf['price_oz']-output_rec.iloc[0]['price_oz']
        price_diff_oz = price_diff_oz.astype('float')
        price_diff_oz = price_diff_oz.values[0]
        # Round similarity, price, and price per oz, and size
        # output_rec.round({'similarity (%)':2,'price ($USD)':2,'price_oz':2,
        # 'size (oz)':2})
        output_rec.round(2)
        output_rec = output_rec[['product_type','brand','product','similarity (%)','price ($USD)', 'price_oz','size (oz)','link']]
        # Header
        st.subheader('Try this product instead:')
        # Make clickable links
        output_rec= output_rec.to_html(render_links=True)
        # Print product recommendation table with clickable link
        st.write(output_rec, unsafe_allow_html=True)
        # st.table(output_rec.style.format({'similarity (%)':'{:.2f}',
        # 'price ($USD)':'{:.2f}','price_oz':'{:.2f}','size (oz)':'{:.2f}'})) 
        st.markdown('')
        st.markdown(f'Savings (total price difference): ${price_diff:.2f}')
        st.markdown(f'Savings (price per oz): ${price_diff_oz:.2f}')
        


