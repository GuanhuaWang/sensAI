from sklearn.cluster.k_means_ import check_random_state, _check_sample_weight, _init_centroids
from sklearn.metrics.pairwise import pairwise_distances_argmin_min, euclidean_distances
from sklearn.utils.extmath import row_norms, squared_norm
import numpy as np


def _labels_inertia(X, sample_weight, x_squared_norms, centers, distances, same_cluster_size=False):
    """E step of the K-means EM algorithm.
    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.
    Parameters
    ----------
    X : float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.
    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.
    x_squared_norms : array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.
    centers : float array, shape (k, n_features)
        The cluster centers.
    distances : float array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.
    Returns
    -------
    labels : int array of shape(n)
        The resulting assignment
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    sample_weight = _check_sample_weight(X, sample_weight)
    n_samples = X.shape[0]

    # Breakup nearest neighbor distance computation into batches to prevent
    # memory blowup in the case of a large number of samples and clusters.
    # TODO: Once PR #7383 is merged use check_inputs=False in metric_kwargs.
    labels, mindist = pairwise_distances_argmin_min(
        X=X, Y=centers, metric='euclidean', metric_kwargs={'squared': True})

    dist = euclidean_distances(X, centers, squared=True)

    print(dist.shape)
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32, copy=False)
    if n_samples == distances.shape[0]:
        # distances will be changed in-place
        distances[:] = mindist
    inertia = (mindist * sample_weight).sum()
    return labels, inertia


def _centers_dense(X, sample_weight, labels, n_clusters, distances):
    """M step of the K-means EM algorithm
    Computation of cluster centers / means.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.
    labels : array of integers, shape (n_samples)
        Current label assignment
    n_clusters : int
        Number of desired clusters
    distances : array-like, shape (n_samples)
        Distance to closest cluster for each sample.
    Returns
    -------
    centers : array, shape (n_clusters, n_features)
        The resulting centers
    """
    # TODO: add support for CSR input
    n_samples = X.shape[0]
    n_features = X.shape[1]

    dtype = np.float32
    centers = np.zeros((n_clusters, n_features), dtype=dtype)
    weight_in_cluster = np.zeros((n_clusters,), dtype=dtype)

    for i in range(n_samples):
        c = labels[i]
        weight_in_cluster[c] += sample_weight[i]
    empty_clusters = np.where(weight_in_cluster == 0)[0]
    # maybe also relocate small clusters?

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            # XXX two relocated clusters could be close to each other
            far_index = far_from_centers[i]
            new_center = X[far_index] * sample_weight[far_index]
            centers[cluster_id] = new_center
            weight_in_cluster[cluster_id] = sample_weight[far_index]

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j] * sample_weight[i]

    centers /= weight_in_cluster[:, np.newaxis]

    return centers


def kmeans_lloyd(X, sample_weight, n_clusters, max_iter=300,
                 init='k-means++', verbose=False, x_squared_norms=None,
                 random_state=None, tol=1e-4, same_cluster_size=False):
    """A single run of k-means, assumes preparation completed prior.
    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.
    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.
    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.
        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.
    tol : float, optional
        The relative increment in the results before declaring convergence.
    verbose : boolean, optional
        Verbosity mode
    x_squared_norms : array
        Precomputed x_squared_norms.
    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.
    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).
    n_iter : int
        Number of iterations run.
    """
    random_state = check_random_state(random_state)
    if same_cluster_size:
        assert len(X) % n_clusters == 0, "#samples is not divisible by #clusters"

    sample_weight = _check_sample_weight(X, sample_weight)
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, sample_weight, x_squared_norms,
                            centers, distances=distances, same_cluster_size=same_cluster_size)

        # computation of the means is also called the M-step of EM
        centers = _centers_dense(
            X, sample_weight, labels, n_clusters, distances)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, sample_weight, x_squared_norms,
                            best_centers, distances=distances, same_cluster_size=same_cluster_size)

    return best_labels, best_inertia, best_centers, i + 1
