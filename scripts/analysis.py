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