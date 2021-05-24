# Data manipulation

import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import numpy as np
import re
import sklearn 

# NLP

import textdistance as td
from sklearn.feature_extraction.text import CountVectorizer

# Machine learning

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

# Plotting

import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'  ")
import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (16, 9.)})
sns.set_style("whitegrid")
sns.set(font_scale = 1.5)

# Miscellaneous

import rootpath
import glob


# Set root path for project

rpath = rootpath.detect()


df_products = pd.read_csv(f"{rpath}/data/clean/data_clean.csv", index_col=0)


df_products.head()


df_products.product_type.unique()


df_products.brand_generic.unique()


df_products.info()


# Aggregate ingredients into lists by product

df_prod_agg = (
    df_products.groupby('product', as_index=False).agg({'ingred_rev':lambda x: list(x)})
)


df_prod_agg.shape


df_prod_agg.info()


# Merge original df on aggregated dataframe to get rest of product information
# Remove product duplicates

df_prod_agg = df_prod_agg.merge(
    df_products.drop('ingred_rev', axis=1), on='product', how='left'
    ) \
    .drop_duplicates(subset=['product'], ignore_index=True) 


df_prod_agg.shape


df_prod_agg.head()


df_prod_agg['product_type'] \
            .value_counts() \
            .plot.pie(autopct='get_ipython().run_line_magic("1.1f%%',", "")
                      title='Relative class frequencies for product types',
                      ylabel='')


df_prod_agg['brand_generic'] \
            .value_counts() \
            .plot \
            .pie(autopct='get_ipython().run_line_magic("1.1f%%',", "")
                 title='Relative class frequencies for ground truth labels',
                 ylabel='')


# Expand ingredient lists per product to a 
# row for each ingredient with repeated rows
# for each product

df_ingred = df_prod_agg.explode('ingred_rev')


# Create a column of 1's for the values to indicate presence of an ingredient

df_ingred['fill_value'] = 1


pivot_indices = (
   ['product','use_category','product_type','brand_generic',
    'brand','size','price','ratings','total_reviews',
    'link']
)


# Fill NAs with "blank" text
# Can't do dropna=False b/c of a memory issue
# NaNs in some of the non-relevant columns of the brand_generics
# So those get dropped from pivoting if NaNs aren't replaced

df_ingred = df_ingred.fillna("blank")


# Pivot table from long to wide 

ingred_counts = pd.pivot_table(df_ingred, values='fill_value',
                                 index=pivot_indices, 
                                 columns='ingred_rev',
                                 fill_value=0,
                                 aggfunc=np.sum,
                                 dropna=True) \
                  .reset_index()       


# Rename column name index to none

ingred_counts.columns.names = [None]


# Function to return training and test set splits 

def get_train_test(target, df, test_size=0.2):
    '''
    Purpose: Return train and test set splits based on the target variable and
    data frame. Since the target is categorical in this case, splits are
    stratified. 
    
    target: the string name of the dataframe column with the target variable
    df: a pandas dataframe holding the features
    '''
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    )
    return X_train, X_test, y_train, y_test


# Split into train and test for prod type

X_train_prodtype, X_test_prodtype, y_train_prodtype, y_test_prodtype = (
    get_train_test('product_type', ingred_counts)
)


# Check class proportions are equal in train and test set

print('Training set relative class frequencies')
y_train_prodtype.value_counts()/len(y_train_prodtype)


print('Test set relative class frequencies')
y_test_prodtype.value_counts()/len(y_test_prodtype)


# Split into train and test for generic labels
# Note test size is 0.5 because there are so few instances of each class!
# Otherwise, for smaller test sizes, we don't get one of each class
# instance in both the training and the test sets

X_train_brandgen, X_test_brandgen, y_train_brandgen, y_test_brandgen = (
    get_train_test('brand_generic', ingred_counts, test_size=0.5)
)


X_train_brandgen.shape


X_test_brandgen.shape


# Check class proportions are equal in train and test set

print('Training set relative class frequencies')
y_train_brandgen.value_counts()/len(y_train_brandgen)


print('Test set relative class frequencies')
y_test_brandgen.value_counts()/len(y_test_brandgen)


y_train_brandgen.head()


# Get product info columns without product

prod_info = [string for string in pivot_indices if string get_ipython().getoutput("= 'product']")


def filter_string(list_of_strings, string_to_remove):
    """
    Purpose: filters a string(s) from a list of strings
    Returns: a list without the indicated string(s)
    string_to_remove: MUST BE A LIST
    """
    new_list = [string for string in list_of_strings \
                    if string not in string_to_remove]
    return new_list


def get_prod_info(df, col_to_remove=[]):
    """
    Purpose: returns ingredient col names and product row names from df 
        to be assigned to tf-idf df
    col_to_remove: list, columns in prod info that are not present in the
        df and need to be removed from prod info since they are not present
        in that df. 
    """
    # Extract ingredient col names
    ingred_cols = df.drop(col_to_remove,
                               axis=1).columns.values
    # Extract product row names
    prod_names = df['product'].values
    return ingred_cols, prod_names


# get column and product names for tf-idf dataframe

brandgen_ingred_names, brandgen_prod_names = (
    get_prod_info(ingred_counts, col_to_remove=prod_info)
)

X_train_brandgen_ingred_names, X_train_brandgen_prod_names = (
    get_prod_info(X_train_brandgen,
                  col_to_remove=filter_string(prod_info,['brand_generic']))
)

X_test_brandgen_ingred_names, X_test_brandgen_prod_names = (
    get_prod_info(X_test_brandgen,
                  col_to_remove=filter_string(prod_info,['brand_generic']))
)


def get_tf(df, col_to_keep):
    """
    Purpose: Get term frequencies from dataframe
    Returns the tf matrix (not dfget_ipython().getoutput(")")
    col_to_keep: List of strings. These should be ingredient column names.
    """
    # Select columns with product name and ingredients only
    # This removes the other columns, e.g. product info 
    df = df[df.columns.intersection(col_to_keep)].set_index(keys='product')
    # convert df to tf matrix
    ingred_tf = df.to_numpy()
    return ingred_tf


def reorder_first_cols(df, col_order):
    '''
    Reorder columns in dataframe with col_order as a list of column names
    in the order you want them in to appear at the beginner of the dataframe.
    The rest of the columns will remain in the same order as before. 
    '''
    # Create new column ordering
    new_col_order = (
          col_order + [col for col in df.columns if col not in col_order]
    )
    # Reindex columns based on new order
    df = df.reindex(columns=new_col_order)
    
    return df


def get_idf(tf_matrix, col_names, prod_names):
    """
    Purpose: Get idf values from matrix of term frequencies.
    Returns df of tf-idf values
    tf_matrix: Term frequency matrix from get_tf() above
    col_names: List of column names for tf_matrix used when constructing df
    prod_names: List of row names for products in the tf-idf df when 
        constructing the tf-idf df
    """
    # Import tfidf transformer
    from sklearn.feature_extraction.text import TfidfTransformer
    # Set transformer arguments 
    transformer = TfidfTransformer(norm='l2',
                               use_idf=True, smooth_idf=True,
                               sublinear_tf=True)
    # Get the TF-IDF weighted ingredient matrix
    tfidf_res = transformer.fit_transform(tf_matrix)
    # Convert to df
    df_tf_idf = pd.DataFrame(tfidf_res.toarray())
    # Add ingredient column names
    df_tf_idf.columns = col_names
    # add product names to df
    df_tf_idf['product'] = prod_names
    # Re-order columns with product being first
    df_tf_idf = reorder_first_cols(df_tf_idf, ['product'])
    return df_tf_idf


def get_tf_idf(df, col_names, prod_names):
    """
    Purpose: get tf-idf matrix from a frequency count table and convert to a
    pandas df. Return the df.
    """
    # Get term frequency matrix for ingredients for brand generic
    ingred_tf = get_tf(df, col_to_keep=col_names)
    # Get idf values 
    df_tf_idf = get_idf(ingred_tf, 
                        col_names=filter_string(col_names, 'product'),  
                        prod_names=prod_names)
    return df_tf_idf


# Get tf-idf values

df_tf_idf = get_tf_idf(ingred_counts,
                       brandgen_ingred_names, 
                       brandgen_prod_names)

df_train_tf_idf = get_tf_idf(X_train_brandgen,
                             X_train_brandgen_ingred_names,
                             X_train_brandgen_prod_names)

df_test_tf_idf = get_tf_idf(X_test_brandgen,
                            X_test_brandgen_ingred_names,
                            X_test_brandgen_prod_names)


# Get unique products in df

prod_original = ingred_counts['product'].values
prod_train = X_train_brandgen['product'].values
prod_test = X_test_brandgen['product'].values


# Subset original long ingredient df for only products in the training set

df_original_ingred = df_ingred.query('product in @prod_original').copy()
df_train_ingred = df_ingred.query('product in @prod_train').copy()
df_test_ingred = df_ingred.query('product in @prod_test').copy()


def add_ord(df):
    """
    Purpose: Create new column with ordinal encodings for ingredient 
    order within each product 
    Returns: dataframe with ordinal encodings
    """
    # Create temporary placeholder constant for ordering
    df['ingred_value'] = 1
    # Add order numbering for ingredients within each product
    df['ingred_order'] = (
        df.groupby('product')['ingred_value'] \
                 .rank(method="first")
    )
    # Count total number of ingredients per product and expand 
    # result into main df
    df = df.assign(
        total_ingred = df.groupby('product')['ingred_order'] \
                                .transform('max')
    )
    # Take difference of ordinal max from ingredient order to get ordinal
    # values for ingredients in decreasing order
    # Add 1 so that the ingredient with the smallest concentration is
    # 1 and not 0 (because 0 would mean it is not present)
    df['ingred_ordinal'] = (
        (df["total_ingred"] - df["ingred_order"])+1
    )
    return df


# Add ordinal weighting to ingredients

df_orginal_ingred_ord = add_ord(df_original_ingred)
df_train_ingred_ord = add_ord(df_train_ingred)
df_test_ingred_ord = add_ord(df_test_ingred)


def scale_ord(df):
    """
    Scale by the number of ingredients in the product so products with 
    different numbers of ingredients are on the same scale
    Minus 1 because added 1 before so last ingredient would not be zero
    """
    df['ingred_ordinal_sc'] = (
        df['ingred_ordinal']/(df["total_ingred"])
    )
    return df


# Scale ordinal encodings

df_orginal_ingred_ord_sc = scale_ord(df_orginal_ingred_ord)
df_train_ingred_ord_sc = scale_ord(df_train_ingred_ord)
df_test_ingred_ord_sc = scale_ord(df_test_ingred_ord)


def add_ord_wts(df_ord_wts, df_tf_idf, df_ingred):
    """
    Purpose: Multiplies scaled ordinal encodings to tf-idf 
    for ingredient values
    Returns: df
    df_ord_wts: df of ingredients with ordinal weightings that
        are also scaled
    df_tf_idf: tf_idf df
    """
    # Convert ordinal weights df from long to wide 
    # with ingredients as the columns and 
    # products as the rows
    df_ingred_ord = df_ord_wts.pivot_table(
                                            index='product',
                                            columns='ingred_rev',
                                            values='ingred_ordinal_sc',
                                            aggfunc='max',
                                            fill_value=0) \
                              .reset_index()   
    # Rename column name index to none
    df_ingred_ord.columns.names = [None]
    # Multiply ingredient tf-idf values by ordinal weightings
    df_ingred_wt = (
        df_tf_idf.drop('product',axis=1) \
                 .mul(df_ingred_ord.drop('product',axis=1),
                      fill_value=0)
    )
    # Add product names and reorder
    df_ingred_wt['product'] = df_tf_idf['product']
    # Extract product information for first row only 
    # (same values for all rows of each product)
    # Use category, price, link, brand, size
    df_prod_info = (
        df_ingred.groupby('product') \
        [['use_category','brand', 'brand_generic',
          'size','price','link']].first().reset_index()
    )
    # Add product info to main ingredient df and reorder
    df_ingred_final = (
        pd.merge(df_ingred_wt, df_prod_info, how='inner',on='product')
    )
    cols_order = ['product', 'use_category','brand','brand_generic',
                  'size','price','link']
    df_ingred_final = reorder_first_cols(df_ingred_final, cols_order)
    return df_ingred_final


# Weight tf-idf values by ordinal weights

df_original_ingred_final = add_ord_wts(df_orginal_ingred_ord_sc,
                                       df_tf_idf,
                                       df_original_ingred)
df_train_ingred_final = add_ord_wts(df_train_ingred_ord_sc,
                                    df_train_tf_idf,
                                    df_train_ingred)
df_test_ingred_final = add_ord_wts(df_test_ingred_ord_sc,
                                   df_test_tf_idf, 
                                   df_test_ingred)


# Import libraries

from sklearn.decomposition import TruncatedSVD
import seaborn as sns


# Create feature matrix

cols_to_drop = ['use_category','brand','brand_generic', 'size','price','link']
df_original_feat = df_original_ingred_final.drop(cols_to_drop,
                                     axis=1).set_index('product')
df_train_feat = df_train_ingred_final.drop(cols_to_drop,
                                     axis=1).set_index('product')
df_test_feat = df_test_ingred_final.drop(cols_to_drop,
                                     axis=1).set_index('product')


# Create a TSVD instance

tsvd = TruncatedSVD(n_components=400, random_state=42)


# Conduct TSVD on features matrix

tsvd_original_res = tsvd.fit(df_original_feat)
df_original_tsvd = pd.DataFrame(tsvd.transform(df_original_feat))

tsvd_train_res = tsvd.fit(df_train_feat)
df_train_tsvd = pd.DataFrame(tsvd.transform(df_train_feat))


def add_prod_info(tsvd_df, original_df, col_names, prod_names_col):
    """
    Purpose: Add product information to TSVD dataframe
    Returns: dataframe with product information in rows and columns
    tsvd_df: df with tsvd results
    original_df: df before tsvd with product information
    col_names: List of strings of column names to be added into tsvd df
    prod_names_col: Name of column with product names, string
    """
    # Add product names
    tsvd_df['product']=original_df[prod_names_col]
    # Copy columns to new df
    col_copy = original_df[col_names].copy()
    # Copy product names to col names df
    col_copy['product'] = original_df[prod_names_col]
    # Join with tsvd df on product
    tsvd_df = pd.merge(tsvd_df, col_copy, how='inner', on='product')
    # Reorder df columns
    cols_order = ['product', 'use_category','brand','brand_generic',
                  'size','price','link']
    tsvd_df = reorder_first_cols(tsvd_df, cols_order)
    return tsvd_df


# Add product info to tsvd df

df_original_tsvd_export = add_prod_info(df_original_tsvd,
                                         df_original_ingred_final,
                                         cols_to_drop,
                                         'product')


# Export tsvd df for analysis

directory_name = '/data/clean/'
file_name = 'data_tsvd_full.csv'
df_original_tsvd_export.to_csv(rpath+directory_name+file_name)








# Plot cumulative explained variance vs. number of components for original

import matplotlib.pyplot as plt
plt.plot(np.cumsum(tsvd_original_res.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# Plot cumulative explained variance vs. number of components for train

import matplotlib.pyplot as plt
plt.plot(np.cumsum(tsvd_train_res.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# Check variance explained at 100 components

tsvd_res.explained_variance_ratio_[0:80].sum()


# Transform test set from training

df_tsvd_test = pd.DataFrame(tsvd_res.transform(df_test_feat))





# Add product info back into SVD df results
df7['product']=df6['product']

# Extract OG product info
product_details=df2[['product','brand','use_subcategory','active','price','size','ratings','total_reviews','link']].copy()

# Rename
product_details=product_details.rename(columns={'use_subcategory':'product_type'})

# price per oz
product_details['price_oz']=(product_details['price']/product_details['size']).round(2)

# Merge product deets with SVD results
df8 = product_details.merge(df7, how = 'left', on = 'product')

# drop duplicates
df8=df8.drop_duplicates(subset = ["product"])

# Reset index
df8.reset_index(drop=True)

# arrange df similarly
df8=df8.sort_values('product')
df6=df6.sort_values('product')

# add vit a
df8['vit_a']=df6['vit_a'].values

#df8.head(n=5)














# Count how many occurences of each ingredient

ingred_cts = df_temp['ingredients'].value_counts().to_frame()


# Turn counts into df and rename columns

ingred_cts = (
    ingred_cts.reset_index() \
    .rename({'index':'ingredient','ingredients':'cts'}, axis = 'columns')
)


ingred_cts.head()


# If active ingredient, paste to other column for text parsing
# df_temp['test'] = np.where(df_temp['ingredients'].str.contains('active')==True,
#                            df_temp['ingredients'],'no_actives')


# If active ingredient, paste to other column for text parsing
# - Make two columns, one for ingredient name, the other for the value
df_temp['active'] = np.where(df_temp['ingredients'].str.contains('active')==True,
                           df_temp['ingredients'],'no_actives')
df_temp['active_value'] = np.where(df_temp['ingredients'].str.contains('active')==True,
                           df_temp['ingredients'],'no_actives')


## Active ingredients
# Remove active ingredient with blanks
df_temp['active']=df_temp['active'].str.replace('.*active.*: |\d*\.*\d*get_ipython().run_line_magic("|\(\d*\.*\d*%\)|", " \d+\.\d+| \d+| +\(sunscreen\)','').str.strip() #|\d*.*\d*%")

# Remove regex error
df_temp['active']=df_temp['active'].str.replace('solar vitis\) and bioactive berry complex','no_actives').str.strip()

# Replace avebenzone
df_temp['active']=df_temp['active'].str.replace('avobenzonem','avobenzone').str.strip()

# Separate clumped actives
df_temp['active']=df_temp['active'].str.replace('avobenzoneoctinoxateoctisalate','no_actives').str.strip() # fix later: avobenzone octinoxate octisalate


## Active ingredient values
# Replace % symbol and remove whitespace
df_temp['active_value']=df_temp['active_value'].str.replace('.*active.*: .* |\(|get_ipython().run_line_magic("\)|\s*%','').str.strip()", "")

# Get rid of non-values
df_temp['active_value']=df_temp['active_value'].str.replace('sunscreen\)|solar vitis\) and bioactive 8 berry complex|acid|no_actives','0').str.strip()

# Correct avobenzone
df_temp['active_value']=df_temp['active_value'].str.replace('active: avobenzonem3.0','3').str.strip()

# Convert to numeric, divide by 100 for proportion, and change NAs to zeros
df_temp['active_value'] = pd.to_numeric(df_temp['active_value'],errors='coerce',downcast='signed').fillna(0)/100


# Label vitamin A 
df_temp['ingredients']=df_temp['ingredients'].str.replace('retinyl palmitate \(vitamin a\)|retinyl palmitate \(vitamin a/vitamine a\)','retinyl palmitate').str.strip()

# Label if present or absent
df_temp['vit_a'] = np.where(df_temp['ingredients'].str.contains('''retinyl palmitate|retinol|bakuchiol|
                                                                        retinyl retinoate|
                                                                        vitamin a \(hpr:hydroxypinacolone retinoate\)'''),
                                'vit_a','no_vit_a')
# Assign value
df_temp['vit_a_value'] = np.where(df_temp['ingredients'].str.contains('''retinyl palmitate|retinol|bakuchiol|
                                                                        retinyl retinoate|
                                                                        vitamin a \(hpr:hydroxypinacolone retinoate\)'''),
                                1,0)


# Create column for presence of ingredient in a product (excluding absences and does not account for all possible ingredients)
df_temp['ingred_value'] = 1
df_temp['skintype_value'] = 1 
df_temp['producttype_value'] = 1 


## Remove common or non-important ingredient rows and ingredient entry errors
# Remove rows that contain the following words or characters
df_temp = df_temp[~df_temp['ingredients'].str.contains('''phenoxyethanol|fragrance|disodium edta|citric acid|
                                                       |xanthan gum|sodium hydroxide|potassium sorbate|sodium benzoate|
                                                       |linalool|carbomer|limonene|sodium chloride|citronellol|
                                                       |geraniol|methylparaben|potassium hydroxide|bht|
                                                       |tetrasodium edta|propylparaben|benzoic acid|trisodium ethylenediamine disuccinate|
                                                       |ethylparaben|\+ plant derived / origine vtale|
                                                       |methylisothiazolinone|opens in a new windo|ulta\.com/coupons for details|
                                                       |see|eugenol|essential oi|glycerin|butylene glycol|caprylyl glycol|
                                                       |ci|ptfe|xantham gum|> denotes organically soure|ethylhexylglycerin|
                                                       |red|green|yellow|blue|organic sucrose \(brown sugar|dimethicone|
                                                       |contains less than 0\.3% thc|f\.i\.l\.# b172461/1|
                                                       |denotes certified organic ingredien|de lagriculture biologique|
                                                       |d227948/1|95% naturally derived naturellement|etc|
                                                       |soothing complex: \[sodium hyaluronate|\+plant derived/origine végétale|
                                                       |lifting phase|phenoxyethnaol|denotes organically sourced|\++|
                                                       |firming phase|with minerals|98% organic of total|\[v2968a|hotheyver|
                                                       |\[v2899a|fd \& c color| denotes organically sourced|5get_ipython().run_line_magic("|\[v2922a|", "")
                                                       |\[v3147a\]|flavor|\(solvent\)|naturally-derived|organi|refer to the product packaging|
                                                       |depending on the location and timing of purchase|napiers moisture formul|
                                                       |xanthangum|xanathan gum|xanthum gum|the most bioavailable form of vitamin c|
                                                       |when skin is overwhelmed due to stress|for the 1st time from vichy|
                                                       |the skin¿s defenses can become overworked|violet|laureth¿4|phenoxyrthanol|
                                                       |67get_ipython().run_line_magic("|lait", " de chèvre\)|\[v2898a|\[v3059a\]|\[v2968a|the carefully selected|")
                                                       |xenthan gum|variations in color|95% naturally derived/dérivé naturellement|
                                                       |de l¿agriculture biologique|# b201629/1''')]
# replace
#'glycerin+|'


ingred_cts = df_test['ingredients'].value_counts()
ingred_cts


ingred_cts = df2['ingredients'].value_counts()
ingred_cts


# Print barplot of counts of each ingredient
import matplotlib.pyplot as plt
#plt.hist(ingred_cts)
#plt.show()

df_test = df_temp['ingredients'].copy().unique()
df_test2 = df_temp['ingredients'].value_counts()
#df_test.columns
#df_test.sort_values('ingred_cts', ascending=False).plot.bar()


# Remove rows with 1 by itself, watch out for removing ceramide 1!
# Replace 2-hexanediol w/ 1,2-hexanediol
# Counts of particular types of ingredients, e.g. extracts, acids, parabens
# heuristic for AHA: glycolic acid, BHA: salicylic acid; malic acid?salicylic acid (0.5get_ipython().run_line_magic(");", " active: salicylic acid (1.0%)")
# What about these? red 40 (ci 16035), yellow 5 (ci 19140)., fd&amp;c yellow no, 5 (ci 19140).
# Vitamin C: ascorbyl palmitate, ascorbic acid
# Vitamin A: retinyl palmitate
# iron oxides (ci 77492).
# active ingredient: salicylic acid (1.51get_ipython().run_line_magic(")", " : if else, paste to other columns")
# other ingredients: water (aqua) : str replace other ingredients
# alcohol denat : str replace with alcohol?


# Code from Jane
# import string

# gamename.translate(str.maketrans('','',string.punctuation))


# Move cleaning data 
df2 = df_temp.copy() 


df2.shape


df2['ingredients'].unique()[:100]


# Add sequence for each ingredient in product
df2['ingred_order'] = df2.groupby('product')['ingred_value'].rank(method="first", ascending=True)

# Get max value of sequence and store in separate df
df2temp = df2.groupby('product')['ingred_order'].max().reset_index()

# Plus 1 so that last ingredient is 1 when take difference of max and ingredient order
# Otherwise zero will indicate that last ingredient isn't present
df2temp['ingred_order'] = df2temp['ingred_order']+1
df2temp=df2temp.rename(columns={"ingred_order":"ingred_ordinal_max"})

# Merge with original DF
df2=pd.merge(df2,df2temp,on='product')

# Take difference of ordinal max from ingredient order to get ordinal values for ingredients
df2['ingred_ordinal'] = df2["ingred_ordinal_max"] - df2["ingred_order"]

# Scale by the ordinal max so products with different numbers of ingredients are on the same scale
# minus 1 because added 1 before so last ingredient would not be zero
df2['ingred_ordinal_sc'] = df2['ingred_ordinal']/(df2["ingred_ordinal_max"]-1)


#df2[df2['brand']=='SUNDAY RILEY']


# Drop page
df2.drop(columns=['page'], inplace=True)


df3 = df2.pivot_table(index=['product','skintype','skintype_value','use_subcategory','producttype_value','active',
                             'active_value'],
                    columns='ingredients',
                    values='ingred_ordinal_sc',
                     aggfunc='max',
                     fill_value=0)
# Put index values back as columns
df3.reset_index(inplace=True)


df3.head()





# Pivot wider based on skintype
df4 = df3.pivot_table(index='product',
                    columns='skintype',
                    values='skintype_value',
                     aggfunc='max',
                     fill_value=0)

# Put index values back as columns
df4.reset_index(inplace=True)


# merge df for one hot encoding for skintypes
df5=pd.merge(df3,df4,on='product')

# Get rid of skintype and skintype_value columns now that they're one hot encoded
df5.drop(columns=['skintype','skintype_value'], inplace=True)


df4_1 = df3.pivot_table(index='product',
                    columns='use_subcategory',
                    values='producttype_value',
                     aggfunc='max',
                     fill_value=0)

# Put index values back as columns
df4_1.reset_index(inplace=True)


# merge df for one hot encoding for skintypes
df5=pd.merge(df5,df4_1,on='product')

# Get rid of skintype and skintype_value columns now that they're one hot encoded
df5.drop(columns=['use_subcategory','producttype_value'], inplace=True)
#df5.head()


# Increase weighting of the active value by adding a constant of 50 plus the active amount
df2['active_value'] = np.where(df2['active_value'] > 0,
                           df2['active_value']+100,df2['active_value'])


df2['active_value'].describe()


df4_2 = df2.pivot_table(index='product',
                    columns='active',
                    values='active_value',
                     aggfunc='max',
                     fill_value=0)

# Put index values back as columns
df4_2.reset_index(inplace=True)


df2.head() 


# merge df for one hot encoding for actives
df5=pd.merge(df5,df4_2,on='product')


# Get rid of skintype and skintype_value columns now that they're one hot encoded
df5.drop(columns=['active','active_value'], inplace=True)
#df5.head()


# Extract distinct rows since no longer need product duplicates
df5=df5.drop_duplicates(subset = ["product"])
#df5.shape


df5.head()


df4_3 = df2.pivot_table(index='product',
                    columns='vit_a',
                    values='vit_a_value',
                     aggfunc='max',
                     fill_value=0)

# Put index values back as columns
df4_3.reset_index(inplace=True)


df4_3.head() 


# merge df for one hot encoding for actives
df5=pd.merge(df5,df4_3,on='product')


df5.head()


# Get rid of skintype and skintype_value columns now that they're one hot encoded
df5.drop(columns=['no_vit_a'], inplace=True)
#df5.head()


df5['vit_a'].values


# Add total ingredients column
tempdf = df2[['ingred_ordinal_max', 'product']]
tempdf = tempdf.rename(columns={'ingred_ordinal_max':'total_ingred'})
# Remove extra 1 added for ordinal encoding
tempdf['total_ingred']=tempdf['total_ingred']-1
# Remove duplicate rows
tempdf=tempdf.drop_duplicates(subset = ["product"])
#tempdf.head()
# Merge df
df6 = df5.merge(tempdf, how = 'left', on = 'product')
df6.head()


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df6[['total_ingred_sc']] = scaler.fit_transform(df6[['total_ingred']])
#df6.head()


df6.head()


#df6.columns[2950:]


## Validation fit for product type

from sklearn.decomposition import TruncatedSVD
#from scipy.sparse import csr_matrix
import seaborn as sns

# Create feature ready df
features = df6.copy().drop(['total_ingred', 'all','face moisturizer','face serum','face wash',
                           'toner','toothpaste'],axis=1).set_index('product')



df6.columns[-30:]


df6.loc[df6['vit_a'] == 1]


df6['vit_a'].values


## FIT SVD FOR SPARSE DATA
# Load libraries

from sklearn.decomposition import TruncatedSVD
#from scipy.sparse import csr_matrix
import seaborn as sns

# Create feature ready df
#features = df6.copy().drop(['total_ingred'],axis=1).set_index('product')
# Try removing features that will wash out the important ingredients
features = df6.copy().drop(['total_ingred','total_ingred_sc','no_actives'],axis=1).set_index('product')


features.shape


# Create a TSVD
tsvd = TruncatedSVD(n_components=20)

# Conduct TSVD on sparse matrix
X_sparse_tsvd = tsvd.fit(features).transform(features)
df7 = pd.DataFrame(X_sparse_tsvd)


# Sum of first three components' explained variance ratios
dat_plot = pd.DataFrame(tsvd.explained_variance_ratio_[0:400])
tsvd.explained_variance_ratio_[0:20].sum()


# Eigenvalues
tsvd.explained_variance_


# Plot 
ax = sns.barplot(x="day", y="tsvd.explained_variance_ratio_[0:2]", data=dat_plot)


import matplotlib.pyplot as plt
plt.plot(np.cumsum(tsvd.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


df7.head()


# Add product info back into SVD df results
df7['product']=df6['product']

# Extract OG product info
product_details=df2[['product','brand','use_subcategory','active','price','size','ratings','total_reviews','link']].copy()

# Rename
product_details=product_details.rename(columns={'use_subcategory':'product_type'})

# price per oz
product_details['price_oz']=(product_details['price']/product_details['size']).round(2)

# Merge product deets with SVD results
df8 = product_details.merge(df7, how = 'left', on = 'product')

# drop duplicates
df8=df8.drop_duplicates(subset = ["product"])

# Reset index
df8.reset_index(drop=True)

# arrange df similarly
df8=df8.sort_values('product')
df6=df6.sort_values('product')

# add vit a
df8['vit_a']=df6['vit_a'].values

#df8.head(n=5)


df8.head()


df6.head()#sort_values('product')


df6['vit_a'].dtype


df8['vit_a']





from sklearn.manifold import MDS

# Create feature ready df
features = df6.copy().drop(['total_ingred'],axis=1).set_index('product')


model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(features)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');


from sklearn import manifold

iso = manifold.Isomap(n_neighbors=6, n_components=2)
iso.fit(features)
manifold_2Da = iso.transform(features)
manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])
manifold_2D


manifold_2D


from sklearn.manifold import TSNE
import seaborn as sns
from bioinfokit.visuz import cluster


df7.head()


# Use df7 that is directly from SVD and doesn't have product added to it
tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300)
tsne_results = tsne.fit_transform(df7)


features_plot = features.copy()


cluster.tsneplot(score=tsne_results)


## PUT TSNE RESULTS IN DF
df_tsne = pd.DataFrame(tsne_results)

# Add product info back into TSNE df results
df_tsne['product']=df6['product']
# Extract OG product info
product_details=df2[['product','brand','use_subcategory','active','price','size','ratings','total_reviews','link']].copy()
# Rename
product_details=product_details.rename(columns={'use_subcategory':'product_type'})
# price per oz
product_details['price_oz']=(product_details['price']/product_details['size']).round(2)
# Merge product deets with SVD results
df_tsne = product_details.merge(df_tsne, how = 'left', on = 'product')
# drop duplicates
df_tsne=df_tsne.drop_duplicates(subset = ["product"])


# Export data for analysis
df_tsne.to_csv(f"{rpath}/data/data_clean.csv",index=True)


color_class = df['class'].to_numpy()
cluster.tsneplot(score=tsne_score, colorlist=color_class, legendpos='upper right', legendanchor=(1.15, 1) )


df8.loc[df8['brand']=='SUNDAY RILEY']


df8


# Export data for analysis
df8.to_csv(f"{rpath}/data/data_clean.csv",index=True)
#df8.shape


# Export data for validation
df8.to_csv(f"{rpath}/data/data_clean_prodtype_valid.csv",index=True)
df8.shape


df8.loc[df8['product_type']=='toothpaste']
