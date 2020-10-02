# import modules
import rootpath
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from mord import LogisticAT
from sklearn.impute import SimpleImputer
from sklearn import metrics
import matplotlib.pyplot as plt

# Set root path for project
path = rootpath.detect()

# Read in data
df = (pd.read_csv(f"{path}/data/data_clean.csv",
                  index_col=False))

df['poor_prop_ingred_cat_rat'].isnull().sum()


df.describe()
df.dtypes

# choose models
from sklearn.linear_model import LinearRegression, LogisticRegression
from mord import LogisticAT
from sklearn.preprocessing import OrdinalEncoder

# instantiate models
model_linear = LinearRegression()
model_1vR = LogisticRegression(multi_class='ovr',
    class_weight='balanced')
model_multi = LogisticRegression(multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced')
model_ordinal = LogisticAT(alpha=0)  # alpha parameter set to zero to perform no regularisation

from sklearn import model_selection
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
import numpy as np

# Impute missing values
df.fillna(df.mean(), inplace=True)

# divide df into features matrix and target vector
features = df.drop(['product_rating', 'Product', 'product_index'], axis=1)
features = df[['good_ingred_cat_rat_n', 'Ingredient_n']]     #df.iloc[:, :-1]  #all except quality
target = df['product_rating']

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, train_size=0.8,test_size=0.2, random_state=1)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# Model score
model.score(X_test, y_test)
r2_score(y_test, predictions)
mean_absolute_error(y_test, predictions)

# 
MAE = make_scorer(mean_absolute_error)
folds = 5
print('Mean absolute error:' )
MAE_linear = cross_val_score(model_linear,
    features,
    target,
    cv=folds,
    scoring=MAE)

# links https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

#https://rikunert.com/ordinal_rating

# define input
new_input = [[6, 30]]
# get prediction for new input
new_output = model.predict(new_input)
print(new_output.round(decimals = 0))

# Plot outputs
plt.scatter(df.Ingredient_n, df.product_rating,  color='black')
plt.savefig('Number of ingredients.png') 
plt.plot(X_test, predictions, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

# Plot EDA
# Number of ingredients in best category
plt.scatter(df.best_ingred_cat_rat_n, df.product_rating,  color='black')
plt.savefig('Number of ingredients in best category.png') 
plt.plot(X_test, predictions, color='blue', linewidth=3)

# Number of ingredients in poor category
plt.scatter(df.poor_ingred_cat_rat_n, df.product_rating,  color='black')
plt.savefig('Number of ingredients in poor category.png') 


plt.xticks(())
plt.yticks(())

plt.show()


plt.hist(df.Prop_labeled, bins = 30)
plt.savefig('PropIngredientsLabeled.png') 
plt.show()
ax = plt.hist(df.Prop_labeled, bins = 30)

ax.set_xlabel("Propotion of ingredients with labeled ratings per product")
ax.set_ylabel("Count")


fig = plt.hist(df.Prop_labeled, bins = 30).get_figure()

fig.savefig('test.pdf')

##### MULTICLASS

#Importing Libraries
import rootpath
import numpy as np
import pandas as pd
from collections import Counter
import re
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import itertools

from ast import literal_eval
# Set root path for project
path = rootpath.detect()

###----
### Food recipe example
df_food = pd.read_json(f"{path}/data/train.json")
df_food.head()

# Extract the features from each recipe (need a global list)
food_features_all_list = []
for i in df_food.ingredients:
    food_features_all_list += i

food_features = list( set(food_features_all_list) )

onehot_ingredients_food = np.zeros((df.shape[0], len(food_features)))

# Index the features (ingredients) alphabetically
feature_lookup_food = sorted(food_features)

# For each recipe look up ingredient position in the sorted ingredient list
# If that ingredient exists, set the appropriate column equal to 1
## This will take 1-2 minutes to finish running
for index, row in df_food.iterrows():
    for ingredient in row['ingredients']:
        onehot_ingredients_food[index, feature_lookup_food.index(ingredient)] = 1

y = df.cuisine.values.reshape(-1,1)
###----

### Mydata
# Read in data
# df = (pd.read_csv(f"{path}/data/data_clean_allingred.csv",
#                   index_col=False))
df = pd.read_csv(f"{path}/data/data_products.csv",
                  index_col=False)

# Clean df ingredients Series
# remove leading and trailing whitespace (but leave whitespace between words) 
#temp_series = pd.Series()
temp_series = []
testx = pd.DataFrame()
for product in df.Ingredients:
    temp_series.append(list(str(product).split(',')))
    for ingredients in product:
        temp_series.append(ingredients.strip())
        pd.concat(product.strip(), temp_series)


df.Ingredients = [[pd.Series(ingredient) for ingredient in str
(product).split(',')] for product in df.Ingredients]

# Make list of each ingredients list with each ingredient as a string
df.Ingredients = [[ingredient for ingredient in str(product).split(', ')] for product in df.Ingredients]

# this does something different with the pd series
# temp_var = [[pd.Series(ingredient) for ingredient in str
# (product).split(',')] for product in df.Ingredients]

# Extracts a list of all ingredients
# Split into ingredients by comma
# nested list of ingredients inside each product
temp_var = [ingredient for product in df.Ingredients for ingredient in str
(product).split(',')]
# temp_var = pd.Series(temp_var)
# remove leading and trailing whitespace (but leave whitespace between words) 
temp_var = temp_var.str.strip()
# Remove colons
temp_var = temp_var.str.replace('^\: ', '')

# Create set of unique ingredients
features_all = list(set(temp_var))

# Create a zeros-only matrix with a row for each recipe and column for each feature
onehot_ingredients = np.zeros((df.shape[0], len(features_all)))

# Index the features (ingredients) alphabetically
feature_lookup = list(sorted(features_all))

# feature_lookup = pd.Series(feature_lookup)

# For each recipe look up ingredient position in the sorted ingredient list
 # If that ingredient exists, set the appropriate column equal to 1
 ## This will take 1-2 minutes to finish running

for product in df:
    for ingredient in df.Ingredients:
        if ingredient in feature_lookup:
            print("yes")


for index, row in df.iterrows():
    for ingredient in row['Ingredients']:
        onehot_ingredients[index, feature_lookup.index(ingredient)] = 1

for index, row in df.iterrows():
    for ingredient in row['Ingredients']:
        onehot_ingredients[index, feature_lookup.index(ingredient)] = 1


# Continuing code

y = df.cuisine.values.reshape(-1,1)

#### OLD

# Create empty list to convert each row of ingredients (string) to a list nested inside the Ingredients series
new = []
for ingredient in df.Ingredients:
    new.append(tuple(ingredient))

df['Ingredients'] = new

# Use b/c set requires a hashable/immutable list
test = []
for i,ingredient in enumerate(df):
    test.append((df.Ingredients[i],))
test = tuple(test)
df['Ingredients'] = test

# iterate with a tuple
test = []
for i,ingredient in enumerate(df.Ingredients):
    test.append((df.Ingredients[i],))
test = tuple(test)
df['Ingredients'] = test

# iterate as a list
test = []
for i,ingredient in enumerate(df.Ingredients):
    test.append(df.Ingredients[i].split(', '))
df['Ingredients'] = test

## First, convert strings to integers, but deal with commas


# Try an iterative list to flatten nested lists
flat_list = [item for sublist in df.Ingredients for item in sublist]
flat_list = []
for i,sublist in enumerate(df.Ingredients):
    for i,item in enumerate(sublist[i]):
        flat_list.append(item[i])

# Convert string lists to integers so you can loop over them
# But issue is dealing with commas and spaces, so the strings can't be converted to floats and then integers
for list in test:
    list.str.split(', ')
    for item in list:
        flat_list.append(item[i].split(', '))




y = [item.split(', ') for item in test]

mylist = [int(x) for x in '3 ,2 ,6 '.split(',')]


temp_df = df.explode('Ingredients')
temp_list = list(itertools.chain(*test.Ingredients))

for list in test:
    for item in int(list):
        flat_list.append(item)


# Create empty list to store product ingredient features
features_all_list = []

# Extract the features from each product (need a global list)
# So for each product, extract the ingredients into a list
for product in df.Ingredients:
    features_all_list += product

test = tuple(map(tuple, features_all_list))

# Extract unique features
features = list(set(features_all_list))

features = list(set(test))

result = sorted(set(map(tuple, features_all_list)), reverse=True)


# divide df into features matrix and target vector
features = df.drop(['product_rating', 'Product', 'product_index'], axis=1)
target = df['product_rating']

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, train_size=0.8,test_size=0.2, random_state=1)



#### TRY THE MODEL

# Read in clean data
df = pd.read_csv(f"{path}/data/temp.csv",
                  index_col=False)

# Import train_test_split
from sklearn.model_selection import train_test_split

# Features
df_features = df.drop('Product', axis=1)

# Target
y = df.Product.values.reshape(-1,1)
# OLD : y = df['Product']

# Check for NAs
df.isnull().sum().sum()

# Split into train, test
X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, shuffle=True, random_state=42)

#### DECISION TREE
# Import decision tree from sklearn
from sklearn.tree import DecisionTreeClassifier
# Set up the decision tree
clf = DecisionTreeClassifier(max_features=2410)
# Fit the decision tree to the training data
clf.fit(X_train, y_train)

# Use the decision tree to predict values for the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score and print the results
a = accuracy_score(y_test, y_pred)
print("Accuracy Score in % : ")
print(a * 100)

## MULTINOMIAL LOGISTIC REGRESSION MODEL 
# import logistic regresion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# Set up and fitlogistic regression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train.ravel())
 
# Get predictions on test data
y_pred = clf.predict(X_test)

# Get accuracy
a = accuracy_score(y_test, y_pred)
print("Accuracy Score in % : ")
print(a * 100)

#### Jaccard similarity

df_products = (pd.read_csv(f"{path}/data/data_products.csv",
                  index_col=False))

# Two similar products, 36, 37
# Two dissimiliar products, 1,2

prod1 = df_products.Ingredients[36].split(',') # split ingredients by commas
prod1 = [element.strip() for element in prod1] # remove whitespace around
prod2 = df_products.Ingredients[37].split(',')
prod2 = [element.strip() for element in prod2] 

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))
list1 = ['dog', 'cat', 'cat', 'rat']
list2 = ['dog', 'cat', 'mouse']

jaccard_similarity(prod1, prod2)

prodlist = prod1 + prod2


#### MCA 
# Tutorial link
# https://github.com/MaxHalford/prince#multiple-correspondence-analysis-mca

import prince

# Read in data
df = pd.read_csv(f"{path}/data/temp.csv",
                  index_col=False)

# Ingredient Features
df_features = df.drop('Product', axis=1)
df_features = df.drop(columns = ['product_index', 'Product', 'product_rating'], axis=1)

# Check for NAs
df_features.isnull().sum().sum()
df.replace([np.inf, -np.inf], np.nan)
df.replace([np.inf, -np.inf], np.nan).dropna(subset=["col1", "col2"], how="all")
np.isinf(df_features).all()

# Check if any values not binary coded by rows
df_features[(df_features >= 2).any(axis=1)]
# nope


# Fit MCA
mca = prince.MCA(
     n_components=2,
     n_iter=5,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42
)

mca = mca.fit(df_features)

## FIT SVD FOR SPARSE DATA
# evaluate svd with logistic regression algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

data = df_features
# define transform
svd = TruncatedSVD()
# prepare transform on dataset
svd.fit(data)
# apply transform to dataset
transformed = svd.transform(data)


# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np
import seaborn as sns

# Make sparse matrix
X_sparse = csr_matrix(data)

# Create a TSVD
tsvd = TruncatedSVD(n_components=100)

# Conduct TSVD on sparse matrix
X_sparse_tsvd = tsvd.fit(data).transform(data)
sparse_df = pd.DataFrame(X_sparse_tsvd)

# Show results
print('Original number of features:', data.shape[1])
print('Reduced number of features:', X_sparse_tsvd.shape[1])

# Sum of first three components' explained variance ratios
tsvd.explained_variance_ratio_[0:200].sum()
dat_plot = pd.DataFrame(tsvd.explained_variance_ratio_[0:200])

# Eigenvalues
tsvd.explained_variance_

# Plot 

ax = sns.barplot(x="day", y="tsvd.explained_variance_ratio_[0:2]", data=dat_plot)

# Add product info back in
sparse_df['product'] = df['Product']

#### K Means

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(sparse_df)

silhouette_score(sparse_df, km.labels_)

# calculate distortion for a range of number of cluster
distortions = []
for i in range(1, 21):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(sparse_df)
    distortions.append(km.inertia_)

plt.plot(range(1, 21), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# plot clusters
y_kmeans = km.predict(sparse_df)
plt.scatter(sparse_df.iloc[:, 0], sparse_df.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# Put cluster labels back into dataset
sparse_df['clusters'] = km.labels_



#### Multinomial test

# Import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Features
mn_features = sparse_df.drop(['clusters'], axis=1)

# Target
y = np.array(sparse_df['clusters'])
y = y.reshape(-1,1)

# Split into train, test
X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=42)

# Set up and fitlogistic regression
clf = LogisticRegression(solver='liblinear',multi_class='ovr').fit(X_train, y_train.ravel())
 
# Get predictions on test data
y_pred = clf.predict(X_test)

# Get accuracy
a = accuracy_score(y_test, y_pred)
print("Accuracy Score in % : ")
print(a * 100)






#### SVD Example
#https://cmdlinetips.com/2019/05/singular-value-decomposition-svd-in-python/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns