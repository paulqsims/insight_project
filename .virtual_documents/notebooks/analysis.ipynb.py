# import modules
import pandas as pd
import numpy as np
import rootpath
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn.metrics as metrics
from sklearn.utils import resample
#import qgrid
import matplotlib.pyplot as plt
import seaborn as sns

# Set root path for project
path = rootpath.detect()

# Read in data
# Use list comprehension to read in all files
df = pd.read_csv(f"{path}/data/data_clean.csv", index_col=0).reset_index(drop=True)


# Product type validation
df = pd.read_csv(f"{path}/data/data_clean_prodtype_valid.csv", index_col=0).reset_index(drop=True)


df[df['brand']=="SUNDAY RILEY"]


#df.head()
df.shape


#features = df.copy().set_index('product')
# Old, before dimensionality reduction
features = df.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
                           'total_reviews','link','price_oz'],
                          axis=1).set_index('product')
#features.head(100)


# Set cluster arguments
kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
        }


silhouette_coefficients = []

# Get silhouette coefficient for each cluster out of 50
for cluster in range(2, 20):
    kmeans = KMeans(n_clusters=cluster, **kmeans_kwargs)
    kmeans.fit(features)
    score = silhouette_score(features, kmeans.labels_)
    silhouette_coefficients.append(score)


plt.style.use("fivethirtyeight")
plt.plot(range(2, 20), silhouette_coefficients)
plt.xticks(range(2, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# Add cluster labels to features



from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(features)
distances, indices = nbrs.kneighbors(features)


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)


dbsc = DBSCAN(eps = 5, min_samples = 5).fit(features)
labels = dbsc.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: get_ipython().run_line_magic("d'", " % n_clusters_)")
print('Estimated number of noise points: get_ipython().run_line_magic("d'", " % n_noise_)")
print("Silhouette Coefficient: get_ipython().run_line_magic("0.3f"", " % metrics.silhouette_score(features, labels))")



from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Plot clustering results

# for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
#     model = AgglomerativeClustering(n_clusters=n_clusters,
#                                     linkage="average", affinity=metric)
#     model.fit(features)
#     plt.figure()
#     plt.axes([0, 0, 1, 1])
#     for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
#         plt.plot(features[model.labels_ == l].T, c=c, alpha=.5)
#     plt.axis('tight')
#     plt.axis('off')
#     plt.suptitle("AgglomerativeClustering(affinity=get_ipython().run_line_magic("s)"", " % metric, size=20)")


# plt.show()

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
y_hc =cluster.fit_predict(features)

#plt.scatter(features[:,1],features[:,2], c=cluster.labels_, cmap='rainbow')
# plt.scatter(features[y_hc ==0,0], features[y_hc == 0,1], s=100, c='red')
# plt.scatter(features[y_hc==1,0], features[y_hc == 1,1], s=100, c='black')
# plt.scatter(features[y_hc ==2,0], features[y_hc == 2,1], s=100, c='blue')
# plt.scatter(features[y_hc ==3,0], features[y_hc == 3,1], s=100, c='cyan')


features['cluster_labels'] = y_hc


features.head()


plt.figure(figsize=(10, 7))  
plt.scatter(features['0'], features['1'], c=features['cluster_labels']) 


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Ingredient Dendograms")
dend = shc.dendrogram(shc.linkage(features, method='ward'))


from sklearn.mixture import GaussianMixture as GMM
#from sklearn.mixture import GMM
gmm = GMM(n_components=4).fit(features)
labels = gmm.predict(features)


plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=labels, s=40, cmap='viridis')


n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(features)
          for n in n_components]

plt.plot(n_components, [m.bic(features) for m in models], label='BIC')
plt.plot(n_components, [m.aic(features) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')


n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(features)
          for n in n_components]

plt.plot(n_components, [m.bic(features) for m in models], label='BIC')
plt.plot(n_components, [m.aic(features) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')


from sklearn.metrics import pairwise_distances


res_pearson = pairwise_distances(features.loc['Gentle Skin Cleanser',:].to_frame().transpose(), features,
                             metric='correlation') #[0:1] .loc["Essential-C Cleanser",:]
res_pearson


res_pearson = res_pearson.reshape(-1)
res_pearson = pd.DataFrame(res_pearson)
res_sim=df[['product','brand','product_type','price','size','ratings',
            'total_reviews','link','price_oz']].copy()
res_sim['similarity']=res_pearson[[0]]
# Round similarity metric
#res_sim['similarity']=round(res_sim['similarity'],2)
# Maybe don't round so you don't have to deal with ties?
#indexNames = res_sim[res_sim['product']=='Essential-C Cleanser'].index
#res_sim.drop(indexNames, inplace=True)
# Sort from top similarity metrics and ignoring self, so starting at 1, not zero
test = res_sim.nlargest(10, 'similarity')[1:10]
#res_sim.head()
# Select top match
test#[:1]


from sklearn.metrics import pairwise_distances
import heapq as hq


res_euc = pairwise_distances(features.loc['Gentle Skin Cleanser',:].to_frame().transpose(), features,
                             metric='euclidean') #[0:1] .loc["Essential-C Cleanser",:]
res_euc


res_euc = res_euc.reshape(-1)
res_euc = pd.DataFrame(res_euc)
res_sim=df[['product','brand','product_type','price','size','ratings',
            'total_reviews','link','price_oz']].copy()
res_sim['similarity']=res_euc[[0]]
# Round similarity metric
#res_sim['similarity']=round(res_sim['similarity'],2)
# Maybe don't round so you don't have to deal with ties?
#indexNames = res_sim[res_sim['product']=='Essential-C Cleanser'].index
#res_sim.drop(indexNames, inplace=True)
# Sort from top similarity metrics and ignoring self, so starting at 1, not zero
test = res_sim.nsmallest(10, 'similarity')[1:10]
#res_sim.head()
# Select top match
test#[:1]


from sklearn.metrics.pairwise import cosine_similarity
import heapq as hq


features.loc['Good Genes All-In-One Lactic Acid Treatment',:].to_frame().transpose()


features.head()


df


#features = df.copy().set_index('product')
# Old, before dimensionality reduction
df_temp = df.copy()

# If active ingredient, paste to other column for text parsing
# - Make two columns, one for ingredient name, the other for the value
df_temp2 = df_temp.loc[df_temp['vit_a']==1]

features = df_temp2.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
                           'total_reviews','link','price_oz'],
                          axis=1).set_index('product')
features.head(10)


features.loc['A+ High-Dose Retinoid Serum',:].to_frame().transpose()


# Calculate cosine similarity for feature 1 
res_cosine = cosine_similarity(features.loc['A+ High-Dose Retinoid Serum',:].to_frame().transpose(), features) #[0:1] .loc["Essential-C Cleanser",:]
res_cosine = res_cosine.reshape(-1)
res_cosine = pd.DataFrame(res_cosine)
res_sim=df[['product','brand','product_type','price','size','ratings', 'active','vit_a',
            'total_reviews','link','price_oz']].copy()
res_sim['similarity']=res_cosine[[0]]
# Round similarity metric
#res_sim['similarity']=round(res_sim['similarity'],2)
# Maybe don't round so you don't have to deal with ties?
#indexNames = res_sim[res_sim['product']=='Essential-C Cleanser'].index
#res_sim.drop(indexNames, inplace=True)
# Sort from top similarity metrics and ignoring self, so starting at 1, not zero
test = res_sim.nlargest(5, 'similarity')[1:5]
#res_sim.head()
# Select top match
test[:10]
#Good Genes All-In-One Lactic Acid Treatment
#A+ High-Dose Retinoid Serum
# Generic vs similar validation
# Cetaphil Daily Facial Cleanser
# Cetaphil Fragrance Free Moisturizing Cream : 0.959282
# Banana Boat Ultra Sport Sunscreen Lotion, Broad Spectum SPF 30 : 0.999971
# St. Ives Fresh Skin Face Scrub, Apricot : 0.999748
# Clean & Clear Essentials Deep Cleaning Toner Sensitive Skin : 0.999994
# Aveeno Positively Radiant Brightening & Exfoliating Face Scrub : 0.990599

#test[test['vit_a']==1]


res_sim.columns


res_cosine = cosine_similarity(features.loc['A+ High-Dose Retinoid Serum',:].to_frame().transpose(), features) #[0:1] .loc["Essential-C Cleanser",:]
res_cosine = res_cosine.reshape(-1)
res_cosine = pd.DataFrame(res_cosine)
res_sim=df[['product','brand','product_type','price','size','ratings', 'active',
            'total_reviews','link','price_oz']].copy()
res_sim['similarity']=res_cosine[[0]]
# Round similarity metric
#res_sim['similarity']=round(res_sim['similarity'],2)
# Maybe don't round so you don't have to deal with ties?

# Sort from top similarity metrics and ignoring self, so starting at 1, not zero
test = res_sim.nlargest(6, 'similarity')[0:6]
#res_sim.head()
# Select top match
test['product_type']#.loc[test['product']=='A+ High-Dose Retinoid Serum',['product_type']]


res_temp = np.where(test['product_type']==test.loc[test['product']=='A+ High-Dose Retinoid Serum',['product_type']].values[0], 1, 0)  


test



# tmp_prodtype = test.loc[test['product']==product,['product_type']].values[0]
# tmp_prodtype
#test['product'].values
for i,product in enumerate(test['product'].values):
    prod_name = test['product'].values[i]
    print(prod_name)
prod_name


test.loc[test['product']==prod_name,['product_type']].values[0]
#tmp_prodtype = test.loc[test['product']==prod_name,['product_type']].values[0]
#tmp_prodtype


test['product_type'].isin(tmp_prodtype).sum()


cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)


product = 'A+ High-Dose Retinoid Serum'
#test.loc[test['product']=='A+ High-Dose Retinoid Serum',['product_type']]
test.loc[test['product']==product,['product_type']].values[0]


test.loc[test['prodtype_match_tot'],1]#.append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)


# Create temp df for troubleshooting for loop
df_temp = df.copy()[:5]
df_temp


# Troubleshooting
# Create features
features = df.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
                           'total_reviews','link','price_oz'],
                          axis=1).set_index('product')


# Troubleshooting for loop

# df to store results
res_sim_prodtype=df[['product','brand','product_type']].copy()

# add empty column for storing results
#res_sim_prodtype['prodtype_match_tot'] = np.nan

prodtype_match_tot = []
#prodtype_match_tot
#

# for each product in df, 
# 1. Get top 6 similiar products, including self
# 2. Count number of products, excluding self, that match input product type
# 3. Add result to df for each product
for i in range(df.shape[0]):
    # Get product name value
    prod_name = df['product'].iloc[i]
    # Calc cosine similarity for the product
    tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
    tmp_cosine = tmp_cosine.reshape(-1)
    tmp_cosine = pd.DataFrame(tmp_cosine)
    # Copy df for storing tmp similarity result
    tmp_sim=df[['product','brand','product_type','price','price_oz']].copy()
    # Add similarity to df
    tmp_sim['similarity']=tmp_cosine[[0]]
    # Sort from top similarity metrics and extract top 6, including self
    tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
    # Store result in res_sim_prodtype df outside for loop
    if prod_name in tmp_top_sim['product']:
        ## Store input product type in tmp_prodtype
        tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
    else: 
        # If the top 6 products do not contain the product entered
        # Select top 5 rows, append the product entered similarity results to the 6th row 
        ## Store input product type in tmp_prodtype
        tmp_top_sim = tmp_top_sim[0:5].append(tmp_sim.loc[tmp_sim['product']==prod_name])
        tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
    ## Check match between input prod type and prod_types in sim output and sum matches
    ## -1 to account for matching with self
    prodtype_match_tot.append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)

# Add for loop results to df
res_sim_prodtype['prodtype_match_tot'] = prodtype_match_tot


res_sim_prodtype.head()


# Get mean proportions
# Empty df to temporarily store for loop rand results in for each iteration
df_temp = pd.DataFrame()
df_temp['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100

#test.append()
#res_sim_prodtype['prodtype_match_tot'].mean()/5
df_temp.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
# test
df_temp.reset_index(level=0, inplace=True)


df_temp.head()


# Extract proportion for one product type
df_temp.at[0,'prodtype_match_tot']


res_sim_prodtype


df_tmp_boot = resample(res_sim_prodtype, replace=True, n_samples=len(res_sim_prodtype))
df_tmp_boot


for product in range(df.shape[0]):
    prod_name = df['product'].iloc[product] 
    print(prod_name)


# Calculate mean/median savings for each product
# i.e. Take the difference in the original price of input product
# and the mean/median price of the top 5 most similar products

# Empty list to store price results in
res_price = []

#for product in range(df.shape[0]):

for product in range(df.shape[0]):
    prod_name = df['product'].iloc[product] 
    ## Empty df to temporarily store for loop rand results in for each iteration
    df_price_temp = pd.DataFrame()
    # Calc cosine similarity for the product
    tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
    tmp_cosine = tmp_cosine.reshape(-1)
    tmp_cosine = pd.DataFrame(tmp_cosine)
    # Copy df for storing tmp similarity result
    tmp_sim=df[['product','brand','product_type','price','price_oz']].copy()
    # Add similarity to df
    tmp_sim['similarity']=tmp_cosine[[0]]
    # Sort from top similarity metrics and extract top 6, including self
    tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
    # Calculate original price
    og_price = res_sim.nlargest(1, 'similarity')['price']
    # Calculate the median price savings
    df_price_temp['med_price_savings']=og_price-tmp_top_sim['price'].median()
    # Calculate the mean price savings
    df_price_temp['mean_price_savings']=og_price-tmp_top_sim['price'].mean()
    # Calculate original price per oz
    og_price_oz = res_sim.nlargest(1, 'similarity')['price_oz']
    # Calculate the median price per oz savings
    df_price_temp['med_price_oz_savings']=og_price_oz-tmp_top_sim['price_oz'].median()
    # Calculate the mean price per oz savings
    df_price_temp['mean_price_oz_savings']=og_price_oz-tmp_top_sim['price_oz'].mean()
    ## clean up
    df_price_temp.reset_index(level=0, inplace=True)
    ## Add iteration identifier
    df_price_temp['iteration'] = product
    ## Append results to tmp list
    res_price.append(df_price_temp)

# Concatenate list results into a df    
df_price_res = pd.concat(res_price, ignore_index=False)
    
# # Create a new copy of dataframe
# df_rand = df.copy()

# # Empty list to store randomization results in
# tmp_list = []

# # Extract features
# features = df.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
#                            'total_reviews','link','price_oz'],
#                           axis=1).set_index('product')

# # For loop for calculating the product type proportion means for each randomization for 1000 iterations
# for iteration in range(1000):
#     # Randomize the product type label
#     df_rand['product_type'] = random.permutation(df['product_type'].values)
#     # create df to store results of product type matches
#     res_sim_prodtype=df[['product','brand','product_type']].copy()
#     # Initialize an empty list to put product type matches in
#     prodtype_match_tot = []
#     # for each product in df, 
#     # 1. Get top 6 similiar products, including self
#     # 2. Count number of products, excluding self, that match input product type
#     # 3. Add result to df for each product
#     for i in range(df_rand.shape[0]):
#         # Get product name value
#         prod_name = df_rand['product'].iloc[i]
#         # Calc cosine similarity for the product
#         tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
#         tmp_cosine = tmp_cosine.reshape(-1)
#         tmp_cosine = pd.DataFrame(tmp_cosine)
#         # Copy df for storing tmp similarity result
#         tmp_sim=df_rand[['product','brand','product_type','price','price_oz']].copy()
#         # Add similarity to df
#         tmp_sim['similarity']=tmp_cosine[[0]]
#         # Sort from top similarity metrics and extract top 6, including self
#         tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
#         # Store result in res_sim_prodtype df outside for loop
#         if prod_name in tmp_top_sim['product']:
#             ## Store input product type in tmp_prodtype
#             tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
#         else: 
#             # If the top 6 products do not contain the product entered
#             # Select top 5 rows, append the product entered similarity results to the 6th row 
#             ## Store input product type in tmp_prodtype
#             tmp_top_sim = tmp_top_sim[0:5].append(tmp_sim.loc[tmp_sim['product']==prod_name])
#             tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
#         ## Check match between input prod type and prod_types in sim output and sum matches
#         ## -1 to account for matching with self
#         prodtype_match_tot.append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)
#     # Add for loop results to df of product type matches
#     res_sim_prodtype['prodtype_match_tot'] = prodtype_match_tot
#     # Create new empty dataframe for mean proportion matches
#     df_prop_prodtype_match = pd.DataFrame()
#     # Add mean proportions for each product type
#     df_prop_prodtype_match['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
#     # Add mean proportion overall
#     df_prop_prodtype_match.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
#     # reset index to make df tidy 
#     df_prop_prodtype_match.reset_index(level=0, inplace=True)
#     ## 
#     # Empty df to temporarily store for loop rand results in for each iteration
#     df_temp = pd.DataFrame()
#     # Calculate mean proportions for each product type 
#     df_temp['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
#     # Calculate overall proportion 
#     df_temp.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
#     # Clean up df
#     df_temp.reset_index(level=0, inplace=True)
#     # Add iteration identifier
#     df_temp['iteration'] = iteration
#     # Append results to tmp list
#     tmp_list.append(df_temp)

# # Concatenate list results into a df    
# df_rand_res = pd.concat(tmp_list, ignore_index=True)
    
# # Concatenate list results into a df    
# df_boot_res = pd.concat(res_boot, ignore_index=True).reset_index()
# df_boot_res['est_type'] = 'Bootstrap'


df_price_res.head()


# Calculate median savings
df_price_res['med_price_savings'].median()


# plot median savings distribution
sns.displot(df_price_res, x="med_price_savings")


# Export savings data
df_price_res.to_csv(f"{path}/data/savings_res.csv",index=True)


# Troubleshooting

# Check variable

range(df_rand.shape[0])


# Calculate mean/median savings for each product
# i.e. Take the difference in the original price of input product
# and the mean/median price of the top 5 most similar products

from numpy import random

# Empty list to store randomization values
rand_res = []

# # Create a new copy of dataframe for storing randomization shuffling
df_rand = df.copy()

for iteration in range(5):
    # Randomize the product 
    df_rand['product'] = random.permutation(df['product'].values)

    # Empty list to store price results in for each randomization iteration
    res_price = []  

    # For each randomization:
    # Calculate the sim scores, price savings for each product
    # DF of sim scores/price savings for each product
    for product in range(df_rand.shape[0]):
        prod_name = df_rand['product'].iloc[product] 
        ## Empty list to temporarily store for loop rand results in for each iteration
        price_temp = []
        # Calc cosine similarity for the product
        tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
        tmp_cosine = tmp_cosine.reshape(-1)
        tmp_cosine = pd.DataFrame(tmp_cosine)
        # Copy df for storing tmp similarity result
        tmp_sim=df[['product','brand','product_type','price','price_oz']].copy()
        # Add similarity to df
        tmp_sim['similarity']=tmp_cosine[[0]]
        # Sort from top similarity metrics and extract top 6, including self
        tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
        # Calculate original price
        og_price = res_sim.nlargest(1, 'similarity')['price']
        # Calculate the median price savings
        med_price_savings=og_price-tmp_top_sim['price'].median()
        # Calculate the mean price savings
        mean_price_savings=og_price-tmp_top_sim['price'].mean()
        # Calculate original price per oz
        og_price_oz = res_sim.nlargest(1, 'similarity')['price_oz']
        # Calculate the median price per oz savings
        med_price_oz_savings=og_price_oz-tmp_top_sim['price_oz'].median()
        # Calculate the mean price per oz savings
        mean_price_oz_savings=og_price_oz-tmp_top_sim['price_oz'].mean()
        ## Add iteration identifier
        product_iter = product
        ## Append results to tmp list
        res_price.append([med_price_savings.values,mean_price_savings.values,
                          med_price_oz_savings.values,mean_price_oz_savings.values,
                         product_iter])

    # For each randomization: #

    # Concatenate for loop list results of price in rand df into a df    
    df_price_res = pd.DataFrame(res_price,
                              columns=['med_price_savings','mean_price_savings',
                                      'med_price_oz_savings','mean_price_oz_savings',
                                      'product_iter'])
    df_price_res = df_price_res.astype(dtype = {"med_price_savings":"float64",
                                                 "mean_price_savings":"float64",
                                                 "med_price_oz_savings":"float64",
                                                 "mean_price_oz_savings":"float64",
                                                 "product_iter":"int64"})

    # Calculate the median price savings of the median price savings across all products for x iteration
    rand_med_price_savings = df_price_res['med_price_savings'].median()

    # Calculate the mean price savings
    rand_mean_price_savings=df_price_res['mean_price_savings'].mean()

    # Calculate the median price per oz savings
    rand_med_price_oz_savings=df_price_res['med_price_oz_savings'].median()

    # Calculate the mean price per oz savings
    rand_mean_price_oz_savings=df_price_res['mean_price_oz_savings'].mean()
    
    # Add iteration identifier
    rand_iteration = iteration
    
    # Append results to tmp list
    rand_res.append([rand_med_price_savings,rand_mean_price_savings,
                    rand_med_price_oz_savings,rand_mean_price_oz_savings,
                    rand_iteration])

# Concatenate for loop list results of price in rand df into a df    
df_rand_price_res = pd.DataFrame(rand_res,
                          columns=['med_price_savings','mean_price_savings',
                                  'med_price_oz_savings','mean_price_oz_savings',
                                  'rand_iter'])

df_rand_price_res = df_rand_price_res.astype(dtype = {"med_price_savings":"float64",
                                                     "mean_price_savings":"float64",
                                                     "med_price_oz_savings":"float64",
                                                     "mean_price_oz_savings":"float64",
                                                     "rand_iter":"int64"})

# # Concatenate for loop list randomized results into a df    
# df_rand_res = pd.concat(rand_res, ignore_index=True)
    
# # Create a new copy of dataframe
# df_rand = df.copy()

# # Empty list to store randomization results in
# tmp_list = []

# # Extract features
# features = df.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
#                            'total_reviews','link','price_oz'],
#                           axis=1).set_index('product')

# # # For loop for calculating the product type proportion means for each randomization for 1000 iterations
# for iteration in range(1000):
#     # Randomize the product type label
#     df_rand['product_type'] = random.permutation(df['product_type'].values)
#     # create df to store results of product type matches
#     res_sim_prodtype=df[['product','brand','product_type']].copy()
#     # Initialize an empty list to put product type matches in
#     prodtype_match_tot = []
#     # for each product in df, 
#     # 1. Get top 6 similiar products, including self
#     # 2. Count number of products, excluding self, that match input product type
#     # 3. Add result to df for each product
#     for i in range(df_rand.shape[0]):
#         # Get product name value
#         prod_name = df_rand['product'].iloc[i]
#         # Calc cosine similarity for the product
#         tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
#         tmp_cosine = tmp_cosine.reshape(-1)
#         tmp_cosine = pd.DataFrame(tmp_cosine)
#         # Copy df for storing tmp similarity result
#         tmp_sim=df_rand[['product','brand','product_type','price','price_oz']].copy()
#         # Add similarity to df
#         tmp_sim['similarity']=tmp_cosine[[0]]
#         # Sort from top similarity metrics and extract top 6, including self
#         tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
#         # Store result in res_sim_prodtype df outside for loop
#         if prod_name in tmp_top_sim['product']:
#             ## Store input product type in tmp_prodtype
#             tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
#         else: 
#             # If the top 6 products do not contain the product entered
#             # Select top 5 rows, append the product entered similarity results to the 6th row 
#             ## Store input product type in tmp_prodtype
#             tmp_top_sim = tmp_top_sim[0:5].append(tmp_sim.loc[tmp_sim['product']==prod_name])
#             tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
#         ## Check match between input prod type and prod_types in sim output and sum matches
#         ## -1 to account for matching with self
#         prodtype_match_tot.append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)
#     # Add for loop results to df of product type matches
#     res_sim_prodtype['prodtype_match_tot'] = prodtype_match_tot
#     # Create new empty dataframe for mean proportion matches
#     df_prop_prodtype_match = pd.DataFrame()
#     # Add mean proportions for each product type
#     df_prop_prodtype_match['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
#     # Add mean proportion overall
#     df_prop_prodtype_match.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
#     # reset index to make df tidy 
#     df_prop_prodtype_match.reset_index(level=0, inplace=True)
#     ## 
#     # Empty df to temporarily store for loop rand results in for each iteration
#     df_temp = pd.DataFrame()
#     # Calculate mean proportions for each product type 
#     df_temp['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
#     # Calculate overall proportion 
#     df_temp.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
#     # Clean up df
#     df_temp.reset_index(level=0, inplace=True)
#     # Add iteration identifier
#     df_temp['iteration'] = iteration
#     # Append results to tmp list
#     tmp_list.append(df_temp)

# # Concatenate list results into a df    
# df_rand_res = pd.concat(tmp_list, ignore_index=True)
    
# # Concatenate list results into a df    
# df_boot_res = pd.concat(res_boot, ignore_index=True).reset_index()
# df_boot_res['est_type'] = 'Bootstrap'


rand_med_price_savings = df_price_res['med_price_savings'].median()
rand_med_price_savings


df_price_res.dtypes


rand_res


df_rand_price_res.head()


 df_price_res.head()


 df_price_res['med_price_savings'].median()


price_rand_temp = []
med_price_savings = df_price_res['med_price_savings'].median()
mean_price_savings = df_price_res['mean_price_savings'].mean()
price_rand_temp.append([med_price_savings, mean_price_savings])


price_rand_temp


df_price_rand_temp = pd.DataFrame(price_rand_temp, columns=['med_price_savings','mean_price_savings'])
df_price_rand_temp


x


df_price_rand_temp['med_price_savings'] = 4


df_price_rand_temp['med_price_savings']


df_price_rand_temp.head()


# Calculate median savings
df_rand_res['med_price_savings'].median()


# plot median savings distribution
sns.displot(df_rand_res, x="med_price_savings")


# Export savings data
df_rand_res.to_csv(f"{path}/data/savings_rand_res.csv",index=True)


# Bootstrap confidence intervals

# Empty list to store boot results in
res_boot = []

for iteration in range(1000):
    # Bootstrap df with replacement 
    df_tmp_boot = resample(res_sim_prodtype, replace=True, n_samples=len(res_sim_prodtype))
    # Get mean proportions
    ## Empty df to temporarily store for loop rand results in for each iteration
    df_boot_temp = pd.DataFrame()
    ## Proportion matched by product type
    df_boot_temp['prodtype_match_tot'] = df_tmp_boot.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
    ## Overall proportion matched
    df_boot_temp.loc['overall'] = df_tmp_boot['prodtype_match_tot'].mean()/5 * 100
    ## clean up
    df_boot_temp.reset_index(level=0, inplace=True)
    ## Add iteration identifier
    df_boot_temp['iteration'] = iteration
    ## Append results to tmp list
    res_boot.append(df_boot_temp)
  
# Concatenate list results into a df    
df_boot_res = pd.concat(res_boot, ignore_index=True).reset_index()
df_boot_res['est_type'] = 'Bootstrap'


# Save bootstrap results
df_boot_res.to_csv(f"{path}/data/boot_prodtype_res.csv",index=True)


# Reimport bootstrap results
df_boot_res = pd.read_csv(f"{path}/data/boot_prodtype_res.csv", index_col=0).reset_index(drop=True)


df_boot_res


df_boot_res.shape


df_boot_res.head()


# Plot distribution of boot results

# Filter toners
df_boot_toner = df_boot_res.query("product_type=='toner'")

# plot toner null distribution
sns.displot(df_boot_toner, x="prodtype_match_tot")


df_boot_toner.head()


# confidence intervals
alpha = 0.05

lower = round(np.quantile(df_boot_toner['prodtype_match_tot'], alpha),2)

upper = round(np.quantile(df_boot_toner['prodtype_match_tot'], (1-alpha)),2)

print(lower,upper)


# Merge boot and randomization data

# Filter toners
df_boot_toner = df_boot_res.query("product_type=='toner'")
df_rand_toner = df_rand_res.query("product_type=='toner'")

df_plot = df_boot_toner.append(df_rand_toner)


df_plot.shape


# Try violin plot
p = sns.violinplot(y = 'product_type',
                   x = 'prodtype_match_tot',
                   hue = 'est_type',
                   inner = None,
                   data = df_plot)
sns.pointplot(y = 'product_type',
              x = 'prodtype_match_tot',
              color = 'black',
              hue = 'est_type',
              join = True,
#              ci = 
              data=df_plot)


# Empty df to store randomization results in
#df_rand_res = pd.DataFrame(columns = ['product_type','prodtype_match_tot','iteration'])
test = []

# Empty df to temporarily store for loop rand results in for each iteration
df_temp = pd.DataFrame()
df_temp['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100

#test.append()
#res_sim_prodtype['prodtype_match_tot'].mean()/5
df_temp.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
# test
df_temp.reset_index(level=0, inplace=True)
df_temp['iteration'] = 1

df_temp
test.append(df_temp)


# Empty df to temporarily store for loop rand results in for each iteration
df_temp = pd.DataFrame()
df_temp['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100

#test.append()
#res_sim_prodtype['prodtype_match_tot'].mean()/5
df_temp.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
# test
df_temp.reset_index(level=0, inplace=True)
df_temp['iteration'] = 2

df_temp
test.append(df_temp)




df_rand_res = pd.concat(test, ignore_index=True)
df_rand_res.head(10)


len(range(5))


# Original df ordering for presentation
df_temp3 = pd.DataFrame(df['product_type'])
df_temp3.head()


# Randomized df ordering for presentation
from numpy import random

df_temp2 = pd.DataFrame(random.permutation(df['product_type'].values))
df_temp2.head()


# Calculate the null probability of product type means

from numpy import random

# Create a new copy of dataframe
df_rand = df.copy()

# Empty list to store randomization results in
tmp_list = []

# Extract features
features = df.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
                           'total_reviews','link','price_oz'],
                          axis=1).set_index('product')

# For loop for calculating the product type proportion means for each randomization for 1000 iterations
for iteration in range(1000):
    # Randomize the product type label
    df_rand['product_type'] = random.permutation(df['product_type'].values)
    # create df to store results of product type matches
    res_sim_prodtype=df[['product','brand','product_type']].copy()
    # Initialize an empty list to put product type matches in
    prodtype_match_tot = []
    # for each product in df, 
    # 1. Get top 6 similiar products, including self
    # 2. Count number of products, excluding self, that match input product type
    # 3. Add result to df for each product
    for i in range(df_rand.shape[0]):
        # Get product name value
        prod_name = df_rand['product'].iloc[i]
        # Calc cosine similarity for the product
        tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
        tmp_cosine = tmp_cosine.reshape(-1)
        tmp_cosine = pd.DataFrame(tmp_cosine)
        # Copy df for storing tmp similarity result
        tmp_sim=df_rand[['product','brand','product_type','price','price_oz']].copy()
        # Add similarity to df
        tmp_sim['similarity']=tmp_cosine[[0]]
        # Sort from top similarity metrics and extract top 6, including self
        tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
        # Store result in res_sim_prodtype df outside for loop
        if prod_name in tmp_top_sim['product']:
            ## Store input product type in tmp_prodtype
            tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
        else: 
            # If the top 6 products do not contain the product entered
            # Select top 5 rows, append the product entered similarity results to the 6th row 
            ## Store input product type in tmp_prodtype
            tmp_top_sim = tmp_top_sim[0:5].append(tmp_sim.loc[tmp_sim['product']==prod_name])
            tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
        ## Check match between input prod type and prod_types in sim output and sum matches
        ## -1 to account for matching with self
        prodtype_match_tot.append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)
    # Add for loop results to df of product type matches
    res_sim_prodtype['prodtype_match_tot'] = prodtype_match_tot
    # Create new empty dataframe for mean proportion matches
    df_prop_prodtype_match = pd.DataFrame()
    # Add mean proportions for each product type
    df_prop_prodtype_match['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
    # Add mean proportion overall
    df_prop_prodtype_match.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
    # reset index to make df tidy 
    df_prop_prodtype_match.reset_index(level=0, inplace=True)
    ## 
    # Empty df to temporarily store for loop rand results in for each iteration
    df_temp = pd.DataFrame()
    # Calculate mean proportions for each product type 
    df_temp['prodtype_match_tot'] = res_sim_prodtype.groupby('product_type')['prodtype_match_tot'].mean()/5 * 100
    # Calculate overall proportion 
    df_temp.loc['overall'] = res_sim_prodtype['prodtype_match_tot'].mean()/5 * 100
    # Clean up df
    df_temp.reset_index(level=0, inplace=True)
    # Add iteration identifier
    df_temp['iteration'] = iteration
    # Append results to tmp list
    tmp_list.append(df_temp)

# Concatenate list results into a df    
df_rand_res = pd.concat(tmp_list, ignore_index=True)


# Save randomization results
df_rand_res.to_csv(f"{path}/data/randomization_res.csv",index=True)


# Reimport randomization results
df_rand_res = pd.read_csv(f"{path}/data/randomization_res.csv", index_col=0).reset_index(drop=True)
df_rand_res['est_type'] = 'Randomization'


df_rand_res.shape


df_rand_res.head()


# Filter face moisturizers
df_rand_moist = df_rand_res.query("product_type=='face moisturizer'")


# Plot distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.displot(df_rand_moist, x="prodtype_match_tot")

# Plot distribution with line for actual estimate
#plt.axvline(df_temp.at[0,'prodtype_match_tot'])
#plt.axvline(df_temp.at[0,'prodtype_match_tot'])


# Filter toners
df_rand_toner = df_rand_res.query("product_type=='toner'")

# plot toner null distribution
sns.displot(df_rand_toner, x="prodtype_match_tot")

# Plot distribution with line for actual estimate
#plt.axvline(df_temp.at[0,'prodtype_match_tot'])


# Filter face washes
df_rand_wash = df_rand_res.query("product_type=='face wash'")

# plot face wash null distribution
sns.displot(df_rand_wash, x="prodtype_match_tot")


# Filter serum
df_rand_serum = df_rand_res.query("product_type=='face serum'")

# plot serum null distribution
sns.displot(df_rand_serum, x="prodtype_match_tot")


# Filter toothpaste
df_rand_toothpaste = df_rand_res.query("product_type=='toothpaste'")

# plot toothpaste null distribution
sns.displot(df_rand_toothpaste, x="prodtype_match_tot")


df_rand_toothpaste


# Calculate p-value for randomization

subset_df = df_rand_toothpaste[df_rand_toothpaste["prodtype_match_tot"] >= df_temp.at[4,'prodtype_match_tot']]
column_count = subset_df.count()

sum(column_count)/1000


# Create df of name brands for checking null distribution of generic matches
nb_scrub_aveeno = 'Aveeno Positively Radiant Brightening & Exfoliating Face Scrub'
nb_toner = 'Clean & Clear Essentials Deep Cleaning Toner Sensitive Skin'
nb_scrub_ives = 'St. Ives Fresh Skin Face Scrub, Apricot'
nb_spf = 'Banana Boat Ultra Sport Sunscreen Lotion, Broad Spectum SPF 30'
nb_cream_cetaphil = 'Cetaphil Fragrance Free Moisturizing Cream'
nb_cleanser_cetaphil = 'Cetaphil Daily Facial Cleanser'


for i in range(len(name_brands)):
        # Get product name value
        prod_name = name_brands[i]
        print(prod_name)


# Calculate the null probability of generic/name brand product means

from numpy import random

# Create a new copy of dataframe
df_rand = df.copy()

# Empty list to store randomization results in
tmp_list = []

# Create list of name brands
name_brands = ['Aveeno Positively Radiant Brightening & Exfoliating Face Scrub',
             'Clean & Clear Essentials Deep Cleaning Toner Sensitive Skin',
             'St. Ives Fresh Skin Face Scrub, Apricot',
             'Banana Boat Ultra Sport Sunscreen Lotion, Broad Spectum SPF 30',
             'Cetaphil Fragrance Free Moisturizing Cream',
             'Cetaphil Daily Facial Cleanser']

# For loop for calculating the product proportion means for each randomization for 1000 iterations
for iteration in range(1000):
    # Randomize the product label of the df_rand but keep everything else the same
    df_rand['product'] = random.permutation(df['product'].values)
    # Extract features from randomized df
    features = df_rand.copy().drop(['product_type','brand', 'price','size','ratings', 'active','vit_a',
                           'total_reviews','link','price_oz'],
                          axis=1).set_index('product')
    # create df to store results of the top 6 product matches from each iteration
    # df should have only 6 rows, one for each name brand
    res_sim_prod=df[df['product'].isin(name_brands)].copy()
    # Initialize an empty list to put product label matches in for each product
    generic_presence = []
    # for each name brand product in list:
    # 1. Get top 6 similiar products, including self
    # 2. Count number of products, excluding self, that match input product type
    # 3. Add result to df for each product
    for i in range(len(name_brands)):
        # Get product name value
        prod_name = name_brands[i]
        # Calc cosine similarities for the product with features from the original, non-product filtered df
        tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
        tmp_cosine = tmp_cosine.reshape(-1)
        tmp_cosine = pd.DataFrame(tmp_cosine)
        # Copy randomized df for storing tmp similarity result
        tmp_sim=df_rand[['product','brand','product_type','price','price_oz']].copy()
        # Add similarity values to temp sim df
        tmp_sim['similarity']=tmp_cosine[[0]]
        # Sort from top similarity metrics and extract top 6, including self
        tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
        # Store result summary for each randomization iteration
        #   in res_sim_prod df outside randomization for loop
        if prod_name in tmp_top_sim['product']:
            ## Store input product label in tmp_prod if top 6 sim products contain product's name
            tmp_prod = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product']].values[0]
        else: 
            # If the top 6 products do not contain the product entered
            # Select top 5 rows, append the product entered similarity results to the 6th row 
            tmp_top_sim = tmp_top_sim[0:5].append(tmp_sim.loc[tmp_sim['product']==prod_name])
            # Store input product type in tmp_prod
            tmp_prod = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product']].values[0]
        # Check match between input name brand product and generic dupe in sim output and sum matches
        #   0 if no matches, and max 1 if a match is present
        # List should be of length 6 after iterating through name_brands list
        if prod_name == 'Aveeno Positively Radiant Brightening & Exfoliating Face Scrub':
            generic_presence.append(tmp_top_sim['product'].str.contains('Beauty 360 Illuminating Facial Scrub').sum())
        elif prod_name == 'Clean & Clear Essentials Deep Cleaning Toner Sensitive Skin':
            generic_presence.append(tmp_top_sim['product'].str.contains('CVS Oil Controlling Astringent Sensitive Skin').sum())
        elif prod_name == 'St. Ives Fresh Skin Face Scrub, Apricot':
            generic_presence.append(tmp_top_sim['product'].str.contains('Mountain Falls Invigorating Apricot Scrub Facial Cleanser').sum())
        elif prod_name == 'Banana Boat Ultra Sport Sunscreen Lotion, Broad Spectum SPF 30':
            generic_presence.append(tmp_top_sim['product'].str.contains('Mountain Falls Active Sport Sunscreen Lotion, SPF 50 Broad Spectrum UVA/UVB Protection').sum())
        elif prod_name == 'Cetaphil Fragrance Free Moisturizing Cream':
            generic_presence.append(tmp_top_sim['product'].str.contains('Amazon Brand - Solimo Ultra Moisturizing Skin Cream for Dry & Sensitive Skin').sum())
        elif prod_name == 'Cetaphil Daily Facial Cleanser':
            generic_presence.append(tmp_top_sim['product'].str.contains('Amazon Brand - Solimo Daily Facial Cleanser, Normal to Oily Skin').sum())
    # Add similarity results of randomization iteration to df of product matches
    res_sim_prod['prod_match_totals'] = generic_presence
    # Create new empty dataframe for product matches
    df_prop_prod_match = pd.DataFrame()
    # Add total matches for each name brand product
    df_prop_prod_match['prop_match'] = res_sim_prod.groupby('product')['prod_match_totals'].sum()
    # Add overall proportion matches across all name brand products
    df_prop_prod_match.loc['overall'] = res_sim_prod['prod_match_totals'].sum()
    # reset index to make df tidy 
    df_prop_prod_match.reset_index(level=0, inplace=True)
    # Add iteration identifier
    df_prop_prod_match['iteration'] = iteration
    # Append results to tmp list
    tmp_list.append(df_prop_prod_match)

# Concatenate list results into a df    
df_rand_res_name_brand = pd.concat(tmp_list, ignore_index=True)


df_rand_res_name_brand.shape


df_rand_res_name_brand.head(24)


# Save randomization results
df_rand_res_name_brand.to_csv(f"{path}/data/randomization_res_namebrands.csv",index=True)


name_brands


# Filter Aveeno
df_rand_aveeno = df_rand_res_name_brand.query("product=='Aveeno Positively Radiant Brightening & Exfoliating Face Scrub'")

# plot Aveeno null distribution
sns.displot(df_rand_aveeno, x="prop_match")


df_rand_aveeno.shape


# Calculate p-value for randomization

df_rand_aveeno["prop_match"].sum()/1000


# Filter Toner
df_rand_tmp = df_rand_res_name_brand.query("product=='Clean & Clear Essentials Deep Cleaning Toner Sensitive Skin'")

# plot Aveeno null distribution
sns.displot(df_rand_tmp, x="prop_match")


# Calculate p-value for randomization

df_rand_tmp["prop_match"].sum()/1000


# Filter St. Ives Fresh Skin Face Scrub, Apricot
df_rand_tmp = df_rand_res_name_brand.query("product=='St. Ives Fresh Skin Face Scrub, Apricot'")

# plot Aveeno null distribution
sns.displot(df_rand_tmp, x="prop_match")


# Calculate p-value for randomization

df_rand_tmp["prop_match"].sum()/1000


# Filter Banana Boat Ultra Sport Sunscreen Lotion, Broad Spectum SPF 30
df_rand_tmp = df_rand_res_name_brand.query("product=='Banana Boat Ultra Sport Sunscreen Lotion, Broad Spectum SPF 30'")

# plot Aveeno null distribution
sns.displot(df_rand_tmp, x="prop_match")


# Calculate p-value for randomization

df_rand_tmp["prop_match"].sum()/1000


# Filter Cetaphil Fragrance Free Moisturizing Cream
df_rand_tmp = df_rand_res_name_brand.query("product=='Cetaphil Fragrance Free Moisturizing Cream'")

# plot Aveeno null distribution
sns.displot(df_rand_tmp, x="prop_match")


# Calculate p-value for randomization

df_rand_tmp["prop_match"].sum()/1000


# Filter Cetaphil Daily Facial Cleanser
df_rand_tmp = df_rand_res_name_brand.query("product=='Cetaphil Daily Facial Cleanser'")

# plot Aveeno null distribution
sns.displot(df_rand_tmp, x="prop_match")


# Calculate p-value for randomization

df_rand_tmp["prop_match"].sum()/1000


# Filter Overall
df_rand_tmp = df_rand_res_name_brand.query("product=='overall'")

# plot Aveeno null distribution
sns.displot(df_rand_tmp, x="prop_match")


# Calculate p-value for randomization
# Also is the overall average for the null distribution
df_rand_tmp["prop_match"].sum()/1000


len(df_rand_toothpaste)


# get mean of null distribution
df_rand_moist.mean()


ordered_df['prodtype_match_tot'].values


# Plot product accuracy: All product types and overall

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

# Set style
sns.set_style("darkgrid")

# Plot horizontal bar plot
#p = sns.barplot(x="prodtype_match_tot", y="product_type", data=test)

# Capitalize product types
test['product_type'] = test['product_type'].str.capitalize()

# Reorder it following the values:
ordered_df = test.reindex([4,3,1,0,2,5]).reset_index(drop=True) # Order of x axis values
my_range=range(1,len(ordered_df.index)+1) #y axis range

plt.hlines(y=my_range, xmin=0, xmax=ordered_df['prodtype_match_tot'], color='black')
plt.plot(ordered_df['prodtype_match_tot'], my_range, "o", color='black')

# Set labels
#p.set(ylabel="Product Type", xlabel="Accuracy: % correct product type match")

plt.xlabel('Product type accuracy: % correct product type match', size = 10, fontsize='large', fontweight='bold',
           color = 'black')
plt.ylabel('', rotation='horizontal', fontsize='large', fontweight='bold', color = 'black')
plt.yticks(my_range, ordered_df['product_type'])
plt.xlim([0,105])

# differentiate toothpaste category
plt.axhline(y = 1.5, ls='--', linewidth=2, color='white', xmin=0, xmax=1)

# Add line differentiating categories from overall
plt.axhline(y = 5.5, ls='-', linewidth=2, color='white', xmin=0, xmax=1)

# Add tight layout so y axis labels don't get cut off
plt.tight_layout()

# Save plot
plt.savefig(f"{path}/plots/product_type_acc_all.png", dpi=300,
           orientation='landscape', pad_inches=0.2)


# Plot product accuracy: All product types and overall

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")

# Plot horizontal bar plot
#p = sns.barplot(x="prodtype_match_tot", y="product_type", data=test)

# Capitalize product types
test['product_type'] = test['product_type'].str.capitalize()

# Create a color if the group is "B"
# my_color=np.where(test['product_type']=='Overall', 'orange',
#                   np.where(test['product_type']=='Toothpaste', 'skyblue', 'black'))
#my_size=np.where(test['product_type']=='Overall', 70,
#                  np.where(test['product_type']=='Toothpaste', 70, 30))

# Reorder it following the values:
ordered_df = test.reindex([4,3,1,0,2,5]).reset_index(drop=True) # Order of x axis values
ordered_df.replace([86.66666667, 97.10843373, 98.56540084, 98.95470383, 99.3258427],0)
my_range=range(1,len(ordered_df.index)+1) #y axis range

plt.hlines(y=my_range, xmin=0, xmax=ordered_df['prodtype_match_tot'], color='black')
plt.plot(ordered_df['prodtype_match_tot'], my_range, "o")

# Set labels
#p.set(ylabel="Product Type", xlabel="Accuracy: % correct product type match")

plt.xlabel('Product type accuracy: % correct product type match', size = 4, fontsize='large', fontweight='bold',
           color = 'black')
plt.ylabel('', rotation='horizontal', fontsize='large', fontweight='bold', color = 'black')
plt.yticks(my_range, ordered_df['product_type'])
plt.xlim([0,105])

# differentiate toothpaste category
plt.axhline(y = 1.5, ls='-', linewidth=2, color='white', xmin=0, xmax=1)

# Add line differentiating categories from overall
plt.axhline(y = 5.5, ls='-', linewidth=2, color='white', xmin=0, xmax=1)

# Add tight layout so y axis labels don't get cut off
plt.tight_layout()

# Save plot
plt.savefig(f"{path}/plots/product_type_acc_overall.png", dpi=1000, figsize=[8, 6],
           orientation='landscape', pad_inches=0.2)


my_size


ordered_df.replace(to_replace = [86.66666667, 97.10843373, 98.56540084, 98.95470383, 99.3258427], value=0, inplace=True)
ordered_df.[ordered_df['prodtype_match_tot'] get_ipython().getoutput("= 98.704545] = 0")


ordered_df


## Calculate average product type accuracy

# df to store results
res_sim_prodtype=df[['product','brand','product_type']].copy()

# add empty column for storing results
#res_sim_prodtype['prodtype_match_tot'] = np.nan

prodtype_match_tot = []
#prodtype_match_tot
#

# for each product in df, 
# 1. Get top 6 similiar products, including self
# 2. Count number of products, excluding self, that match input product type
# 3. Add result to df for each product
for i in range(df.shape[0]):
    # Get product name value
    prod_name = df['product'].iloc[i]
    # Calc cosine similarity for the product
    tmp_cosine = cosine_similarity(features.loc[prod_name,:].to_frame().transpose(), features)
    tmp_cosine = tmp_cosine.reshape(-1)
    tmp_cosine = pd.DataFrame(tmp_cosine)
    # Copy df for storing tmp similarity result
    tmp_sim=df[['product','brand','product_type','price','price_oz']].copy()
    # Add similarity to df
    tmp_sim['similarity']=tmp_cosine[[0]]
    # Sort from top similarity metrics and extract top 6, including self
    tmp_top_sim = tmp_sim.nlargest(6, 'similarity')[0:6]
    # Store result in res_sim_prodtype df outside for loop
    ## Store input product type in tmp_prodtype
    tmp_prodtype = tmp_top_sim.loc[tmp_top_sim['product']==prod_name,['product_type']].values[0]
    ## Check match between input prod type and prod_types in sim output and sum matches
    ## -1 to account for matching with self
    prodtype_match_tot.append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)
    
    #res_sim_prodtype.loc[res_sim_prodtype['prodtype_match_tot']].append(tmp_top_sim['product_type'].isin(tmp_prodtype).sum()-1)
    


top_sim = res_sim.nlargest(6, 'similarity')[1:6]
og_price = res_sim.nlargest(1, 'similarity')['price']
#best_sim_score = min(max(top_sim['similarity'] + min(top_sim['price_oz'])))
#best_sim_score
#test=top_sim.iloc[0].to_frame().transpose()[['brand','product','price','price_oz','size','link']]
og_price


og_price-top_sim['price'].median()


test


### TEST STREAMLIT AREA
prod_type = 'face wash'


df.columns


df2 = df[df.product_type==(f'{prod_type}')]


df2.brand.unique()


tempdf = df.loc[df['product']==f'{product}']
df.loc[df['product']==f'{product}']


price_diff = tempdf['price_oz']-output_rec['price_oz']


tempdf = df.loc[df['product']==f'{product}']
test.iloc[1]['price_oz']
#price_diff = tempdf['price_oz']-test['price_oz']
#price_diff=price_diff.astype('float')
#price_diff
#tempdf['price_oz']
#price_diff
#test=price_diff.values[0]
#test
#f"test price:{test}"


output_rec = top_sim.iloc[0].to_frame().transpose()[['product_type', 'brand','product','price','price_oz','size','link']]
output_rec


top_sim


top_sim.nsmallest(1, 'price')


output_rec = top_sim.nsmallest(1, 'price')
output_rec


tempdf = df.loc[df['product']==f'{product}']
res_sim
#tempdf
price_diff = tempdf['price_oz']-output_rec['price_oz']
#price_diff
#res_sim['price_oz']
#price_diff
#tempdf['price_oz']#-res_sim['price_oz']
top_sim = res_sim.nlargest(6, 'similarity')[1:6]
output_rec = top_sim.iloc[1].to_frame().transpose()[['product_type', 'brand','product', 'similarity','price','price_oz','size','link']]
output_rec['similarity']=output_rec['similarity'].astype(float)
output_rec['similarity']=round(output_rec['similarity'],2)
#output_rec
#tempdf = df.loc[df['product']==f'{product}']
#tempdf
output_rec['similarity']


product_input=df.loc[df['product']==f'{product}']
product_input


cosine_similarity(product_input) 


top_sim[:1]


(((0.881751+0.766394+0.629398)/3) + ((0.629398+0.526911+0.539059)/3) +
((0.881751+0.757364+0.526911)/3) + ((0.766394+0.757364+0.539059)/3))/4


test
#np.minimum(test['similarity'],test['price_oz'])
min(max(test.similarity),min(test.price_oz))


features.index
#features.loc["Essential-C Cleanser",:]  #"'Buffet' + Copper Peptides 1get_ipython().run_line_magic(""", "")
features.loc["Essential-C Cleanser",:].to_frame().transpose()


features.loc["Essential-C Cleanser",:].to_frame().transpose()
#features.loc[0,["Essential-C Cleanser"]]


#res_cosine
res_cosine.nlargest(6, 0)[1:6]


features.loc["Essential-C Cleanser":]


#df.loc[df['brand']=='The Ordinary']
#df.loc[df['brand']=='CeraVe']
#df.loc[df['brand']=='Kate Somerville'] # INDIE LEE
#df.loc[df['brand']=='Walgreens']
#df.loc[df['brand']=='SUNDAY RILEY']
df.loc[df['product_type']=='toothpaste']
#df.loc[df['product_type']=='toner']
#df['brand'].unique()


df.loc[df['brand']=='SUNDAY RILEY'] #and df['brand']=='Peter Thomas Roth'



