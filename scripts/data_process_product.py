# import modules
import rootpath
import pandas as pd
import numpy as np

# Set root path for project
path = rootpath.detect()

# Read in data
## Product information
df_raw = pd.read_csv(f"{path}/data/rating1_scrape.csv",
 index_col=0)

df_list = []
for df in range(1,6):
    (df_list.append(pd.read_csv(f"{path}/data/rating{df}_scrape.csv",
                    index_col=0)))
df_product_ratings_raw = pd.concat(df_list)

# Make a copy of the dfs
df_prod_rat = df_product_ratings_raw.copy()

# Examine df
df_prod_rat.head()
df_prod_rat.describe()
df_prod_rat.dtypes

# Convert ratings to lower case
df_prod_rat['Ingredients'] = df_prod_rat['Ingredients'].str.lower().astype('str')
df_prod_rat['Rating'] = df_prod_rat['Rating'].str.lower()
df_prod_rat['Product'] = df_prod_rat['Product'].str.lower()

# Remove categories: from Category text
df_prod_rat['Ingredients'] = df_prod_rat['Ingredients'].str.replace('\n\tIngredients\n\n\t\n\t\:', '')

df_prod_rat['Ingredients'].head()

df_prod_rat['Ingredients'] = df_prod_rat['Ingredients'].str.replace('\\n(.*)\\t', '')

df_prod_rat['Ingredients'] = df_prod_rat['Ingredients'].astype('str').str.replace(\\(.*)\\t', '')

temp_df = []
temp_df['que'] = np.where((df_ingred_cat['Ingredient'] = df_prod_rat['Ingredients'].str.contains(df_ingred_cat['Ingredient'])), df_ingred_cat['Ingredient'])


import re

test = re.sub('\\n(.*?)\\t', '',df_prod_rat['Ingredients'].iloc[0])
test

test = []
test = df_prod_rat['Ingredients'].str.replace('\\n\\tIngredients\\n\\n\\t\\n\\t\\t', 'TEST', regex=True)
test[:10]
df_prod_rat['Ingredients'].str.contains('\\n(.*)\\t', regex=True)


# Isolate rating number
# Removes text in pattern matches and replaces with nothing
pattern = '|'.join(['<span class="product-rating rating stars-', '"></span>'])
df_prod_rat['Rating'] = df_prod_rat['Rating'].str.replace(pattern, '')
