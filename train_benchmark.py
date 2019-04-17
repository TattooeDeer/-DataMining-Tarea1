# General purpose
import numpy as np
from scipy.spatial.distance import chebyshev
import pandas as pd
from collections import Counter
import string
from copy import copy
from tqdm import tqdm

# joblib
from joblib import dump

# Graphical
import matplotlib.pyplot as plt
import seaborn as sn


# Seed
seed = 15051991 # My birthday because I love myself <3

# Joblib backend
from joblib import parallel_backend

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score

# Fuzzy clustering
from skfuzzy.cluster import cmeans, cmeans_predict

import warnings
warnings.filterwarnings('ignore')


def train_estimator(estimator, df, grid = None, verbose = True, cv = 5):
    predicts = []

    best_score = -1
    # No idea how to dynamically generate a matrix from a variable number of parameters
    # to explore so I will hardcode this to infinity and beyond
    if estimator == KMeans:
        print('############### K-Means ###############')
        best_estimator = GridSearchCV(KMeans(), param_grid = grid, cv = cv, verbose = 2, n_jobs = -1).fit(df)
        predicts = best_estimator.predict(df)
        print('\t\t Silhouette Score: {}'.format(silhouette_score(df, predicts)))
        print('\t\t Nº Clusters: {}'.format(best_estimator.best_params_['n_clusters']))

    elif estimator == AgglomerativeClustering:
        if verbose:
            print('############### Agglomerative Clustering ###############')
        #linkage_list = ['ward', 'single', 'complete', 'average']
        for link in tqdm(grid['linkage']) :
            print('Linkage: {}'.format(link))
            best_score_link = -1
            for n_clusters in grid['n_clusters']:
                agg_predicts =AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(df) 
                if len(pd.Series(agg_predicts).unique()) == 1:
                        continue
                sil_score = silhouette_score(df, agg_predicts)
                if sil_score > best_score_link:  # If its the best Ive seen keep it
                    best_n_clusters_link = n_clusters
                    best_score_link = sil_score
                    predicts_link = agg_predicts
                if sil_score > best_score:  # If its the best Ive seen keep it
                    best_n_clusters = n_clusters
                    best_link = link
                    best_score = sil_score
                    predicts = agg_predicts
                    best_estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(df)
                    if verbose:
                        print('* New best global combination of params found:')
                        print('\t\t Silhouette Score: {}'.format(best_score))
                        print('\t\t Nº Clusters: {}'.format(best_n_clusters))
                        print('\t\t Linkage: {}'.format(best_link))
            print('Best Silhouette Score for {0} Link:\nSilhouette Score = {1}\nNº clusters = {2}'.format(link, best_score_link,
                                                                                                         best_n_clusters_link))
        print('Best Global Silhouette Score:\nLink = {0}\nSilhouette Score = {1}\nNº clusters = {2}'.format(
                                                                                                            best_link,
                                                                                                            best_score,
                                                                                                            best_n_clusters))
    elif estimator == SpectralClustering:
        if verbose:
            print('############### Spectral Clustering ###############')
        for n_clusters in tqdm(grid['n_clusters']):
                spectral_predicts = SpectralClustering(n_clusters=n_clusters, n_jobs = -1).fit_predict(df) 
                if len(pd.Series(spectral_predicts).unique()) == 1:
                        continue
                sil_score = silhouette_score(df, spectral_predicts)
                if sil_score > best_score:  # If its the best Ive seen keep it
                    best_n_clusters = n_clusters
                    best_score = sil_score
                    predicts = spectral_predicts
                    best_estimator = SpectralClustering(n_clusters=n_clusters, n_jobs = -1).fit(df) 
                    if verbose:
                        print('* New best combination of params found:')
                        print('\t\t Silhouette Score: {}'.format(best_score))
                        print('\t\t Nº Clusters: {}'.format(best_n_clusters))
    elif estimator == DBSCAN:
        if verbose:
            print('############### DBSCAN ###############')
        eps_range = np.linspace(1, 10, dtype = int)
        min_samples_range = np.linspace(5, 100, 20, dtype = int)
        metrics = ['manhattan','euclidean', 'cosine', chebyshev]
    
        for metric in tqdm(grid['metric']):
            for min_sample in grid['min_samples']:
                for eps in grid['eps']:
                    dbscan_predicts = DBSCAN(metric = metric, min_samples = min_sample, eps = eps,
                                            n_jobs = -1).fit_predict(df)
                    if len(pd.Series(dbscan_predicts).unique()) == 1:
                        continue
                    try:
                        sil_score = silhouette_score(df, dbscan_predicts)
                        if sil_score > best_score:
                            best_metric = metric
                            best_min_samples = min_sample
                            best_eps = eps
                            best_score = sil_score
                            best_estimator = DBSCAN(metric = best_metric, min_samples = best_min_samples,
                                                    eps = best_eps, n_jobs = -1).fit(df)
                            predicts = dbscan_predicts
                            if verbose:
                                print('* New best combination of params found:')
                                print('\t\t Silhouette Score: {}'.format(best_score))
                                print('\t\t Min Samples: {}'.format(best_min_samples))
                                print('\t\t EPS: {}'.format(best_eps))
                                print('\t\t Metric: {}'.format(best_metric))
                    except ValueError:
                        print('Warning: DBSCAN found only one cluster, silhouette score needs a number of clusters between '\
                            '2<= n_clusters <=n_samples-1 to be calculated.\nReturning the predicted labels...')
                        # predicts = DBSCAN(metric = metric, min_samples = min_sample, eps = eps,
                        #                 n_jobs = -1).fit_predict(df)
                        return predicts, DBSCAN(metric = metric, min_samples = min_sample, eps = eps,
                                            n_jobs = -1).fit(df)
    elif estimator == MeanShift:
        if verbose:
            print('############### MeanShift ###############')
        #bandwidth_range = range(1,10,1)
        #cluster_all_options = [True, False]
        for cluster_all_option in tqdm(grid['cluster_all']):
            for bandwidth in grid['bandwidth']:
                meanshift_predicts = MeanShift(bandwidth = bandwidth, cluster_all = cluster_all_option,
                                               n_jobs = -1).fit_predict(df)
                if len(pd.Series(meanshift_predicts).unique()) == 1:
                        continue
                sil_score = silhouette_score(df, meanshift_predicts)
                if sil_score > best_score:
                    best_bandwidth = bandwidth
                    best_cluster_all = cluster_all_option
                    best_score = sil_score
                    predicts = meanshift_predicts
                    best_estimator = MeanShift(bandwidth = bandwidth, cluster_all = cluster_all_option,
                                               n_jobs = -1).fit(df)
                    if verbose:
                        print('* New best combination of params found:')
                        print('\t\t Silhouette Score: {}'.format(best_score))
                        print('\t\t Bandwidth: {}'.format(best_bandwidth))
                        print('\t\t Cluster all: {}'.format(best_cluster_all))
    elif estimator == cmeans:
        if verbose:
            print('############### Fuzzy C-Means ###############')
        bandwidth_range = range(1,10,1)
        cluster_all_options = [True, False]
        for nclusters in grid['c']:
            for m in grid['m']:
                meanshift_predicts = cmeans_predict(bandwidth = bandwidth, cluster_all = cluster_all_option,
                                               n_jobs = -1).fit_predict(df)
                if len(pd.Series(meanshift_predicts).unique()) == 1:
                        continue
                sil_score = silhouette_score(df, meanshift_predicts)
                if sil_score > best_score:
                    best_bandwidth = bandwidth
                    best_cluster_all = cluster_all_option
                    best_score = sil_score
                    predicts = meanshift_predicts
                    best_estimator = MeanShift(bandwidth = bandwidth, cluster_all = cluster_all_option,
                                               n_jobs = -1).fit(df)
                    if verbose:
                        print('* New best combination of params found:')
                        print('\t\t Silhouette Score: {}'.format(best_score))
                        print('\t\t Bandwidth: {}'.format(best_bandwidth))
                        print('\t\t Cluster all: {}'.format(best_cluster_all))
    return predicts, best_estimator

def plot_cluster(predicts, df):
    '''In: A list of labels and a df
       Out: A sexy plot
    '''
    df_plot = copy(df)
    plt.rc('font', size = 16)
    plt.rcParams['figure.figsize'] = 12, 12
    sn.set_style('white')
    sn.set_palette(sn.color_palette('colorblind'))

    alphabet = list(string.ascii_lowercase)[0:len(pd.Series(predicts).unique())]
    if len(pd.Series(predicts)) > len(list(string.ascii_lowercase)): # Demasiadas etiquetas, solo plotear colores
        df_plot['label'] = predicts
        sn.despine()
        #fig, ax = plt.subplots(figsize=(11, 11))
        sn.lmplot(data = df_plot, x = 'x', y = 'y', hue = 'label', fit_reg = False,
                                 legend = False, legend_out = False)
        
        #plt.show(cluster_plot)
    else:
        df_plot['label'] = pd.Series(predicts).replace(list(pd.Series(predicts).unique()),
                                                    list(string.ascii_uppercase))
        
        sn.despine()
        #fig, ax = plt.subplots(figsize=(11, 11))
        sn.lmplot(data = df_plot, x = 'x', y = 'y', hue = 'label', fit_reg = False,
                                 legend = True, legend_out = True)
        
        #plt.show(cluster_plot)

def plot_cluster_interactive(data, estimator, **kwargs):
    if estimator == KMeans:
        preds = KMeans(kwargs['n_clusters']).fit_predict(data)

    elif estimator == AgglomerativeClustering:
        preds = AgglomerativeClustering(n_clusters=kwargs['n_clusters'], linkage=kwargs['linkage']).fit_predict(data) 
    
    elif estimator == SpectralClustering:
        preds = SpectralClustering(n_clusters=kwargs['n_clusters'], n_jobs = -1).fit_predict(data)

    elif estimator == DBSCAN:
        preds = DBSCAN(metric = kwargs['metric'], min_samples = kwargs['min_samples'], eps = kwargs['eps'],
                                            n_jobs = -1).fit_predict(data)
    
    elif estimator == MeanShift:
        preds = MeanShift(bandwidth = kwargs['bandwidth'], cluster_all = kwargs['cluster_all'],
                                               n_jobs = -1).fit_predict(data)
    plot_cluster(preds, data)

    
def create_benchmark(estimator, data, grid, cv = 5, verbose = True):
    estimator_preds, estimator_fitted = train_estimator(estimator = estimator, df = data, grid = grid, cv = cv)
    plot_cluster(estimator_preds, data)
    return estimator_fitted
    