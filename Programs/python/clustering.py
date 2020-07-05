from sklearn.cluster import AgglomerativeClustering
import h5py as h5
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from plda.my_func import lennorm


def ahc_clustering(unlabeled_file, labeled_file, n_clusters=10, display=False):
    """
    Perform agglomerative clustering on the unlabeled file (in .h5 format)
    and produce a labeled file (in .h5 format).
    :param unlabeled_file:
    :param labeled_file:
    :param n_clusters:
    :return:
    """
    with h5.File(unlabeled_file) as f:
        X = f['X'][:]
        n_frames = f['n_frames'][:]
        spk_path = f['spk_path'][:]

    ahc = AgglomerativeClustering(linkage='complete', n_clusters=n_clusters,
                                  affinity='cosine')
    ahc.fit(lennorm(X))

    os.remove(labeled_file) if os.path.isfile(labeled_file) else None
    unicode = h5.special_dtype(vlen=str)
    with h5.File(labeled_file, 'w') as f:
        f['X'] = X
        f['n_frames'] = n_frames
        f['spk_path'] = spk_path
        spk_ids = []
        for label in ahc.labels_:
            spk_ids.append('spk-' + str(label))
        f['spk_ids'] = np.array(spk_ids, dtype=unicode)

    if display:
        plot_dendrogram(ahc, labels=ahc.labels_)
        plt.show(block=False)
        lbs, lbs_count = np.unique(ahc.labels_, return_counts=True)
        plt.bar(lbs, lbs_count)
        plt.xlabel('Cluster Index')
        plt.ylabel('No. of Samples')
        plt.show()


# https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
# Authors: Mathew KalladaAuthors:
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)











