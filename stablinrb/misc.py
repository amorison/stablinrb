import numpy as np
import matplotlib.pyplot as plt


def savefig(fig, name):
    fig.savefig(name, format='PDF', bbox_inches='tight')
    plt.close(fig)


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

    The chosen value is the component of modes[norm_mode] with
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


def build_slices(i0n, imax):
    """Build slices from a list of min/max index

    i0n: list of (imin, imax) indices tuples, one tuple per variable,
         without the boundaries if they are not needed
    imax: maximum index taking boundaries into account typically the
          number of Chebyshev points)
    """
    igf = [lambda idx: idx - i0n[0][0]]
    i_0s, i_ns = zip(*i0n)
    for i_0, i_n in zip(i_0s[1:], i_ns):
        ipn = igf[-1](i_n)
        i_g = lambda idx, i_l=ipn-i_0+1: idx + i_l
        igf.append(i_g)
    slall = []  # entire vector
    slint = []  # interior points
    slgall = []  # entire vector with big matrix indexing
    slgint = []  # interior points with big matrix indexing
    for i_0, i_n, i_g in zip(i_0s, i_ns, igf):
        slall.append(slice(i_0, i_n + 1))
        slint.append(slice(1, imax))
        slgall.append(slice(i_g(i_0), i_g(i_n + 1)))
        slgint.append(slice(i_g(1), i_g(imax)))
    return i0n, igf, slall, slint, slgall, slgint
