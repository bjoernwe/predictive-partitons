import numpy as np
import scipy.linalg
import scipy.sparse.linalg


def entropy(x, normalize=False, ignore_empty_classes=True):
    """
    Calculates the (normalized) entropy over a given probability 
    distribution.
    """

    # invalid input?
    assert x.ndim == 1
    assert True not in list(x < -0.)
    assert float("nan") not in x
    assert float("inf") not in x

    # useful variables
    trans_sum = np.sum(x)
    K = len(x)

    # only one class?
    if K <= 1:
        return 1.0

    # empty class?
    if trans_sum == 0:
        if not ignore_empty_classes:
            print "Warning: Entropy of zero vector. Returning max. entropy."
        if normalize:
            return 1.0
        else:
            return np.log2(K)

    # the actual calculation
    probs = np.array(x, dtype=np.float32) / trans_sum
    log_probs = np.zeros_like(probs)
    log_probs[probs > 0.] = np.log2( probs[probs > 0.] )
    entropy = -np.sum(probs * log_probs)

    # normalization?
    assert entropy <= np.log2(K) + 1e-6
    if normalize:
        entropy /= np.log2(K)

    assert(entropy >= 0)
    return entropy


def entropy_rate(P, mu=None, normalize=False):
    """
    Calculates the entropy rate for a given transition matrix, i.e. the 
    transition entropy of each state weighted by its stationary distribution
    mu.
    """
    
    # valid input?
    assert P.ndim == 2        
    N, M = P.shape
    assert N == M
    
    # mu
    if mu is None:
        Q = P + 1e-6
        d = np.sum(Q, axis=1)
        Q = Q / d[:,np.newaxis]
        if N <= 2:
            E, U = scipy.linalg.eig(Q.T)
            idx = np.argsort(E.real)
            assert abs(E[idx[-1]].real - 1) < 1e-6
            mu = U[:,idx[-1]].real
        else:
            E, U = scipy.sparse.linalg.eigs(Q.T, k=1, which='LR')
            assert abs(E[0].real - 1) < 1e-6
            mu = U[:,0].real
        
    # normalize mu
    assert mu.ndim == 1
    mu /= np.sum(mu)
    
    # row entropies
    row_entropies = np.zeros(N)
    for i in range(N):
        if mu[i] > 0:
            row_entropies[i] = entropy(P[i], normalize=normalize)

    # weighted average
    h = np.sum(mu * row_entropies)
    return h


# def mutual_information(P, mu=None):
#     """
#     Calculates the mutual information between t and t+1 for a model given
#     as transition matrix P. 
#     """
#     
#     # valid inuput?
#     if P is None:
#         return None
#     assert P.ndim == 2
#     assert np.sum(P) > 0
#     
#     # dimensionality
#     N, M = P.shape
#     assert N == M
#     
#     # prepare stationary distribution mu
#     if mu is None:
#         mu = np.ones(N) / N
#     else:
#         assert mu.ndim == 1
#         assert True not in list(mu < -0.)
#         assert float("nan") not in mu
#         assert float("inf") not in mu
#         mu_sum = np.sum(mu)
#         assert mu_sum >= 0
#         mu /= np.sum(mu_sum)
#         
#     # the actual calculation
#     h_mu = entropy(mu)
#     h_p = entropy_rate(P, mu=mu)
#     mi = h_mu - h_p
#     return mi

    

if __name__ == '__main__':
    pass