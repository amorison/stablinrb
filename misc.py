import numpy as np


def normalize(arr, norm=None):
    """Normalize complex array with element of higher modulus

    if norm is given, a first normalization with this norm is done
    """
    if norm is not None:
        arr = arr / norm
    amax = arr[np.argmax(np.abs(arr))]
    return arr / amax, amax


def normalize_modes(modes, norm_mode=3, full_norm=True):
    """Normalize modes

    Since modes are eigenvectors, one can choose an arbitrary
    value to normalize them without any loss of information.

    The chose value is the component of modes[norm_mode] with
    the max modulus.
    All the other modes are then normalized by their maximum
    modulus component.
    This function return the set of modes after those two
    normalizations, and the set of normalization components.
    """
    ref_vector, norm = normalize(modes[norm_mode])
    normed_vectors = []
    norm_values = []
    for ivec in range(0, norm_mode):
        if full_norm:
            nvec, nval = normalize(modes[ivec], norm)
        else:
            nvec, nval = modes[ivec] / norm, 1
        normed_vectors.append(nvec)
        norm_values.append(nval)
    normed_vectors.append(ref_vector)
    norm_values.append(norm)
    for ivec in range(norm_mode + 1, len(modes)):
        if full_norm:
            nvec, nval = normalize(modes[ivec], norm)
        else:
            nvec, nval = modes[ivec] / norm, 1
        normed_vectors.append(nvec)
        norm_values.append(nval)
    return normed_vectors, norm_values
