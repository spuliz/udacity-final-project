# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans

# Learning Curves, to have a look at Bias/Variance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
# from plot_learning_curve import plot_learning_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold


def fixDataset(df):
    """
    Description:
        Convert features CAMEO_DEUG_2015 CAMEO_INTL_2015 as there are same values but different types (int and float) to float
        Replace X and XX with np.nan. Encode to binary 1-0 the values W and O in OST_WEST_KZ column. Create dummy columns out of    categorical variables.
        Drops useless and redundant columns.
    INPUT:
    df - (pandas dataframe) df dataframe general population, customers or mailout
    
    OUTPUT:
    df - dataframe with fixed features   
    """
    print('fix index')
    df = df.set_index(['LNR'])
    print('fixing cameo columns...')
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].replace({'X': np.nan})
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype(float)
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].replace({'XX':np.nan})
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].astype(float)
    
    # CAMEO_DEU_2015
    cameo_deu = pd.get_dummies(df['CAMEO_DEU_2015'], prefix='CAMEO_DEU_2015', dummy_na=False).astype('int64')
    df = pd.concat([df, cameo_deu], axis=1)
    
    print('converting to binary OST_WEST_KZ column...')
    dict_ost_west = {'OST_WEST_KZ': {'W':0, 'O':1}}
    df = df.replace(dict_ost_west)
    
    print('fixing D19_LETZTER_KAUF_BRANCHE...') #assign an int to categorical values   
    d19 = pd.get_dummies(df['D19_LETZTER_KAUF_BRANCHE'], prefix='D19_LETZTER_KAUF_BRANCHE', dummy_na=False)
    d19 = d19.astype('int64')
    df = pd.concat([df, d19], axis=1)
    
    print('dropping useless columns...')
    col_to_drop = ['CAMEO_DEU_2015','EINGEFUEGT_AM', 'D19_LETZTER_KAUF_BRANCHE'] 
    df.drop(col_to_drop, axis=1, inplace=True)

    return df



#function to deal with all the missing and unknown entries
def unknowns_to_NANs(df, xls):
    '''
    This function uses the information in the Dias files to help identify values that are missing or unknown
    Replaces missing or unknown value with nan
    '''
    
    for _, row in xls[xls['Meaning'].isin(['unknown', 'unknown / no main age detectable'])].iterrows():
        key, val = row['Attribute'], dict()
        for x in row['Value'].split(', '):
            val[int(x)] = np.nan

        if key in df.columns:
            df = df.replace({key: val})
    return df



def drop_null_columns(df, columns_to_drop):
    """
    Description:
        Drop columns with null values (calculated from the population dataset) and LNR column.
    
    INPUT:
        df (dataframe): population or customer dataframe
    
    OUTPUT:
        df (dataframe): dataset without dropped columns 
    """
    print("dropping columns..")

    df.drop(columns_to_drop, axis=1, inplace=True)
    
    print("Done")
    
    return df


def clean_dataset(df, columns_to_drop, customer_df=False, mailout_df = False):
    """
    DESCRIPTION:
        Wrapper function for all cleaning steps performed
    
    INPUT:
        df (dataframe): population, customer or mailout dataframe
    
    OUTPUT:
        df (dataframe): cleaned dataset  
    """
    if customer_df == True:
        df = df.drop(labels=['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'], axis=1)
    if 'Unnamed: 0' in df:
        df.drop('Unnamed: 0', axis = 1, inplace = True)
    df = fixDataset(df)
    if mailout_df == False:
        df = df.drop_duplicates()
    df = drop_null_columns(df, columns_to_drop)
    print('Done!')
    return df

def transform_dataset(df, imputer):
    """
        DESCRIPTION: A function that impute null values (simple imputer) and scale data
    
        INPUT: a pandas dataframe
    
        OUTPUT: a transformed pandas dataframe 
    """
    print("imputing...")
    df_imputed = pd.DataFrame(imputer.fit_transform(df))
    print("Done!")
    df_imputed.columns = df.columns
    df_imputed = df_imputed.set_index(df.index)
    return df_imputed

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
def show_pca(pca):
    """
    DESCRIPTION:
        Visualize the curves of the explained variance ratio for each component and the cumulative ratios
    
    INPUT:
        pca: an sklearn.decomposition.pca.PCA object    
    
    OUTPUT:
        None (visualizes a plot)
    
    """
    cumulative_ratios=np.zeros(len(pca.explained_variance_ratio_))
    for i in range(len(pca.explained_variance_ratio_)):
        cumulative_ratios[i]=np.sum(pca.explained_variance_ratio_[:i])
    plt.figure(figsize=(20,10))
    plt.plot(pca.explained_variance_ratio_)
    plt.plot(cumulative_ratios)
    plt.xlabel("Components")
    plt.ylabel("Explained Variance Ratio %")
    plt.title("PCA Components Explained Variance Ratios")
    plt.yticks(np.arange(0, 1, step=0.05))
    plt.xticks(np.arange(0, len(pca.explained_variance_ratio_)+2, step= (len(pca.explained_variance_ratio_) // 20)))
    plt.grid(linewidth=0.1)
    plt.legend(['Variance Ratio', 'Cumulative'], loc='center right')
    

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
# Apply PCA with enough components to explain 90% of the variance ratio
def apply_pca(df, n_components=300):
    """
    DESCRIPTION:
        Apply PCA to the customers or general population dataset
    INPUT:
        df (dataframe): a cleaned and imputed dataset (customers or gen population)
        n_components (number): the number of components we wish to retain
    OUTPUT:
        df_pca (dataframe): a dataframe where the columns are the latent features from the PCA
        pca: the sklearn.decomposition.pca.PCA object (fitted)
    """
    
    pca = PCA(n_components=n_components)
    pca.fit(df)
    df_pca = pca.transform(df)
    
    return df_pca, pca

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
def pca_results(full_dataset, pca, component):
    """
    DESCRIPTION:
        Create a DataFrame of the PCA results
        Includes dimension feature weights and explained variance
        Visualizes the PCA results
    INPUT:
        full_dataset: the population or customer dataset before applying the PCA
        pca: the sklearn.decomposition.pca.PCA object (fitted)
        component (integer) : the component we want to visualize the features and weights
    OUTPUT:
    
    """

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (20,10))

    # Plot the feature weights as a function of the components
    features_to_show = components.iloc[component - 1].sort_values(ascending=False)
    features_to_show = features_to_show[np.absolute(features_to_show.values) >= 0.01]
    components.iloc[component - 1].sort_values(ascending=False).plot(ax = ax, kind = 'bar');
    features_to_show.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(list(features_to_show.keys()), rotation=90)
    ax.set_xlabel("Features")

    # Display the explained variance ratios
    #for i, ev in enumerate(pca.explained_variance_ratio_):
    ev = pca.explained_variance_ratio_[component-1]
    
    plt.title("Component {} Explained Variance: {:.2f}%".format(component, ev*100))

    return features_to_show

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
# Distance in two dimensions
def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    
    """
    DESCRIPTION: 
        Calculate Euclidean distance for each data point assigned to centroid (in two dimensions)
    INPUT:
        data: 
        cx: x coordinate of the centrod
        cy: y coordinate of the centroid
        i_centroid: centroid (number)
        cluster_labels: cluster labels associated to data points (predicted)
    
    OUTPUT:
        mean of the distances of the data points from the centroid
    
    """
    # Calculate Euclidean distance for each data point assigned to centroid 
    distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
    # return the mean value
    return np.mean(distancess)

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
# Distance in n-dimensions
def k_mean_distance_n(data, centroid, i_centroid, cluster_labels):
    """
       DESCRIPTION: 
        Calculate Euclidean distance for each data point assigned to centroid (in n dimensions)
    INPUT:
        data: dataset (usually with latent features)
        centroid: array (n_dimensions) of the centroid coordinate
        i_centroid: centroid (number)
        cluster_labels: cluster labels associated to data points (predicted)
    
    OUTPUT:
        mean of the distances of the data points from the centroid
    """
    
    # Calculate Euclidean distance for each data point assigned to centroid 
    distances = [np.linalg.norm(x - centroid) for x in data[cluster_labels == i_centroid]]
    # return the mean value
    return np.mean(distances)

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
# compute the average within-cluster distances.
def k_means_score(data, n_clusters):
    """
    DESCRIPTION:
        Apply K-Means clustering to the dataset. Calculate the mean distance of the points of each cluster from its centroid
        and return the mean of all means (across all clusters)
    
    INPUT:
        data: dataset (usually with latent features)
        n_clusters: number of clusters to apply K-Means
    
    OUTPUT:
        the mean of the distances of the datapoints of each cluster from his centroid  
    """
    # Run K-Means
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=20000)
    #kmeans = KMeans(n_clusters=n_clusters)
    model = kmeans.fit(data)
    labels = model.predict(data)
    centroids = model.cluster_centers_
    
    total_distance = []
    for i, c in enumerate(centroids):
        # Function from above
        mean_distance = k_mean_distance_n(data, c, i, labels)
        total_distance.append(mean_distance)
    return(np.mean(total_distance))

# credit: https://github.com/santamm/Customer-Segmentation/blob/master/Arvato%20Project%20Workbook.ipynb
def apply_KMeans(df_pca, n_clusters):
    """
    DESCRIPTION:
        Apply K-Means clustering to the dataset with a given number of clusters.
    
    INPUT:
        df_pca: dataset (usually with latent features)
        n_clusters: number of clusters to apply K-Means
    
    OUTPUT:
        gen_labels: labels (cluster no) for each data point
        gen_centroids: the list of coordinate of each centroid
        k_model (sklearn.cluster.k_means_.MiniBatchKMeans ): the cluster model 
    """
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=50000, random_state=3425)
    kmeans_model = kmeans.fit(df_pca)
    pop_labels = kmeans_model.predict(df_pca)
    pop_centroids = kmeans_model.cluster_centers_
    
    return pop_labels, pop_centroids, kmeans_model


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



