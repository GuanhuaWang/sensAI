import numpy as np

from even_k_means import kmeans_lloyd


def kmeans_grouping(features, targets, n_groups, same_group_size=True, seed=42):
    class_indices = targets.unique().sort().values
    mean_vectors = []
    for t in class_indices:
        mean_vec = features[targets == t.item(), :].mean(dim=0)
        mean_vectors.append(mean_vec.cpu().numpy())
    X = np.asarray(mean_vectors)
    class_indices = class_indices.cpu().numpy()
    assert X.ndim == 2
    best_labels, best_inertia, best_centers, _ = kmeans_lloyd(
        X, None, n_groups, verbose=True,
        same_cluster_size=same_group_size,
        random_state=seed,
        tol=1e-6)
    groups = []
    for i in range(n_groups):
        groups.append(class_indices[best_labels == i].tolist())
    return groups


def group_classes(n_groups, all_features, all_targets, seed, group_selection_algorithm='even_k_means'):
    if group_selection_algorithm == 'even_k_means':
        groups = kmeans_grouping(all_features, all_targets,
                                 n_groups, same_group_size=True, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm '{group_selection_algorithm}'")
    return groups
