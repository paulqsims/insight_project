# import modules
import rootpath
import pandas as pd
import numpy as np

# Set root path for project
path = rootpath.detect()

# Read in data
## Ingredient information
df_ingredient_raw = pd.read_csv(f"{path}/data/ingredient_cat_scrape.csv",
 index_col=0, nrows=1832)

## Product information
df_raw = pd.read_csv(f"{path}/data/rating1_scrape.csv",
 index_col=0)

df_list = []
for df in range(1,6):
    (df_list.append(pd.read_csv(f"{path}/data/rating{df}_scrape.csv",
                    index_col=0)))
df_product_ratings = pd.concat(df_list)

df_product_ratings.head()
df_product_ratings.describe()





# Make a copy of the df
df_ingred_cat = df_ingredient_raw.copy()

# Examine df
df_ingred_cat.head()
df_ingred_cat.describe()
df_ingred_cat.dtypes

# Convert ratings to lower case
df_ingred_cat['Ingredient'] = df_ingred_cat['Ingredient'].str.lower()
df_ingred_cat['Rating'] = df_ingred_cat['Rating'].str.lower()
df_ingred_cat['Category'] = df_ingred_cat['Category'].str.lower()

# Remove categories: from Category text
df_ingred_cat['Category'] = df_ingred_cat['Category'].str.replace('categories:', '')
