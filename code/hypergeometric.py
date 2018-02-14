"""Run python3 hypergeometric.py from the command line to run the unit tests"""

from __future__ import division
import math
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats
from scipy.optimize import minimize_scalar
import itertools

### Tri-hypergeometric distribution tests

def diluted_margin_trihypergeometric(w, l, n, N_w, N_l, N):
    """
    Conduct tri-hypergeometric test
    
    H_0: N_w - N_l <= c
    H_1: N_w - N_l > c
    
    using the diluted margin as test statistic.
    Parameters
    ----------
    w : int
        number of votes for w in sample
    l : int
        number of votes for l in sample
    n : int
        number of ballots in the sample
    N_w : int
        total number of votes for w in the population *under the null*
    N_l : int
        total number of votes for l in the population *under the null*
    N : int
        total number of ballots in the population
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n.
    """
    N_u = N-N_w-N_l
    pairs = itertools.product(range(n+1), range(n+1))   # Cartesian product
    pairs = itertools.filterfalse(lambda y: sum(y) > n or y[0] - y[1] < (w-l), pairs)
    pairs = itertools.filterfalse(lambda y: y[0] > N_w or y[1] > N_l or n-y[0]-y[1] > N_u, pairs)
    return sum(map(lambda p: sp.special.comb(N_w, p[0], exact=True)*\
                         sp.special.comb(N_l, p[1], exact=True)*\
                         sp.special.comb(N_u, n-p[0]-p[1], exact=True),\
                         pairs))/sp.special.comb(N, n, exact=True)


def diluted_margin_trihypergeometric2(w, l, n, N_w, N_l, N):
    """
    Conduct tri-hypergeometric test
    
    H_0: N_w - N_l <= c
    H_1: N_w - N_l > c
    
    using the diluted margin as test statistic.
    Parameters
    ----------
    w : int
        number of votes for w in sample
    l : int
        number of votes for l in sample
    n : int
        number of ballots in the sample
    N_w : int
        total number of votes for w in the population *under the null*
    N_l : int
        total number of votes for l in the population *under the null*
    N : int
        total number of ballots in the population
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n.
    """
    pvalue = 0
    N_u = N-N_w-N_l
    for ww in range(w-l, n+1):
        tmp = 0
        for ll in range(0, ww-w+l+1):
            if ww+ll > n:
                break
            else:
                tmp += sp.misc.comb(N_l, ll)*sp.misc.comb(N_u, n-ww-ll)
        pvalue += tmp * sp.misc.comb(N_w, ww)
    return pvalue/sp.misc.comb(N, n)


def trihypergeometric_optim(sample, popsize, null_margin):
    '''
    Wrapper function for p-value calculations using the tri-hypergeometric distribution.
    This function maximizes the p-value over all possible values of the nuisance parameter,
    the number of votes for the reported winner in the population.
    
    Parameters
    ----------
    sample : array-like
        sample of ballots. Values must be 0 (votes for l), 1 (votes for w), and np.nan (other votes).
    popsize : int
        total number of ballots in the population
    null_margin : int
        largest difference in *number* of votes between the reported winner and reported loser,
        N_w - N_l, under the null hypothesis
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n.
    '''
    
    w = sum(sample==1)
    l = sum(sample==0)
    n = len(sample)
    u = n-w-l

    # maximize p-value over N_w
    optim_fun = lambda N_w: diluted_margin_trihypergeometric(w, l, n, N_w, N_w-null_margin, popsize)
    # conditions are that N_w+N_l = 2*upper - c < N-u, N_l = upper-c > l, N_w = upper > w
    upper_Nw = int((popsize-u+null_margin)/2)
    lower_Nw = int(np.max([w, null_margin]))
    return np.max(list(map(optim_fun, range(lower_Nw, upper_Nw+1))))


### Hypergeometric tests

def diluted_margin_hypergeometric(w, l, N_w, N_l):
    """
    Conduct hypergeometric test
    
    H_0: N_w - N_l <= c
    H_1: N_w - N_l > c
    
    using the diluted margin as test statistic.
    The test conditions on n and w+l.
    
    Parameters
    ----------
    w : int
        number of votes for w in sample
    l : int
        number of votes for l in sample
    N_w : int
        total number of votes for w in the population *under the null*
    N_l : int
        total number of votes for l in the population *under the null*
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n and w+l.
    """
    n = w+l
    pvalue = sp.stats.hypergeom.sf(w-1, N_w + N_l, N_w, n)
    return pvalue


def diluted_margin_hypergeometric2(w, l, N_w, N_l):
    """
    Conduct hypergeometric test
    
    H_0: N_w - N_l <= c
    H_1: N_w - N_l > c
    
    using the diluted margin as test statistic.
    The test conditions on n and w+l.
    
    Parameters
    ----------
    w : int
        number of votes for w in sample
    l : int
        number of votes for l in sample
    N_w : int
        total number of votes for w in the population *under the null*
    N_l : int
        total number of votes for l in the population *under the null*
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n and w+l.
    """
    pvalue = 0
    delta = w-l
    n = w+l
    for ww in range(int((delta+n) / 2), n+1):
        pvalue += sp.stats.hypergeom.pmf(ww, N_w + N_l, N_w, n)
    return pvalue


def diluted_margin_hypergeometric3(w, l, N_w, N_l):
    """
    Conduct hypergeometric test
    
    H_0: N_w - N_l <= c
    H_1: N_w - N_l > c
    
    using the diluted margin as test statistic.
    The test conditions on n and w+l.
    
    Parameters
    ----------
    w : int
        number of votes for w in sample
    l : int
        number of votes for l in sample
    N_w : int
        total number of votes for w in the population *under the null*
    N_l : int
        total number of votes for l in the population *under the null*
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n and w+l.
    """
    delta = w-l
    n = w+l
    pairs = itertools.product(range(n+1), range(n+1))
    pairs = itertools.filterfalse(lambda y: sum(y) != n, pairs)
    pairs = itertools.filterfalse(lambda y: y[0] - y[1] < delta, pairs)
    
    pvalue = 0
    for p in pairs:
        pvalue += sp.stats.hypergeom.pmf(p[0], N_w + N_l, N_w, p[0]+p[1])
    return pvalue


def hypergeometric_optim(sample, popsize, null_margin):
    '''
    Wrapper function for p-value calculations using the hypergeometric distribution.
    This function maximizes the p-value over all possible values of the nuisance parameter,
    the number of votes for the reported winner in the population.
    
    Parameters
    ----------
    sample : array-like
        sample of ballots. Values must be 0 (votes for l), 1 (votes for w), and np.nan (other votes).
    popsize : int
        total number of ballots in the population
    null_margin : int
        largest difference in *number* of votes between the reported winner and reported loser,
        N_w - N_l, under the null hypothesis
    Returns
    -------
    float
        conditional probability, under the null, that difference in the
        number of votes for candidate w and the number of votes for candidate l,
        divided by the sample size n, will be greater than or equal to (w-l)/n.
        The test conditions on n and w+l.
    '''
    
    w = sum(sample==1)
    l = sum(sample==0)
    n = len(sample)
    u = n-w-l    

    # maximize p-value over N_w
    optim_fun = lambda N_w: diluted_margin_hypergeometric(w, l, N_w, N_w-null_margin)
    # conditions are that N_w+N_l = 2*upper - c < N-u, N_l = upper-c > l, N_w = upper > w
    upper_Nw = int((popsize-u+null_margin)/2)
    lower_Nw = int(np.max([w, null_margin]))
    
    return np.max(list(map(optim_fun, range(lower_Nw, upper_Nw+1))))


### Unit tests

def test_find_pairs_trihyper():
    # example: w=2, l=1, n=3
    pairs = itertools.product(range(3+1), range(3+1))
    pairs = itertools.filterfalse(lambda y: sum(y) > 3, pairs)
    pairs = itertools.filterfalse(lambda y: y[0] - y[1] < (2-1), pairs)
    expected_p = [(1, 0), (2, 0), (2, 1), (3, 0)]
    assert list(pairs)==expected_p
    
    # example: w=4, l=1, n=5
    pairs = itertools.product(range(5+1), range(5+1))
    pairs = itertools.filterfalse(lambda y: sum(y) > 5, pairs)
    pairs = itertools.filterfalse(lambda y: y[0] - y[1] < (4-1), pairs)
    expected_p = [(3, 0), (4, 0), (4, 1), (5, 0)]
    assert list(pairs)==expected_p
    
    
def test_diluted_margin_pvalue_trihyper():
    # example 1: w=2, l=1, n=3, W=L=U=2
    t1 = 2*1*1/sp.misc.comb(6, 3) # w=1, l=0, u=2
    t2 = 1*1*2/sp.misc.comb(6, 3) # w=2, l=0, u=1
    t3 = 1*2*1/sp.misc.comb(6, 3) # w=2, l=1, u=0
    t4 = 0                        # w=3, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric(2, 1, 3, 2, 2, 6), t1+t2+t3+t4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric2(2, 1, 3, 2, 2, 6), t1+t2+t3+t4)
    
    # example 2: w=4, l=1, n=5, W=5, L=U=2
    t1 = sp.misc.comb(5, 3)*1*1/sp.misc.comb(9, 5) # w=3, l=0, u=2
    t2 = sp.misc.comb(5, 4)*1*2/sp.misc.comb(9, 5) # w=4, l=0, u=1
    t3 = sp.misc.comb(5, 4)*2*1/sp.misc.comb(9, 5) # w=4, l=1, u=0
    t4 = 1*1*1/sp.misc.comb(9, 5)                  # w=5, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric(4, 1, 5, 5, 2, 9), t1+t2+t3+t4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric2(4, 1, 5, 5, 2, 9), t1+t2+t3+t4)
    
    # example 3: w=2, l=0, n=4, W=3, L=1, N=6. Result should be 0.4
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric(2, 0, 4, 3, 1, 6), 0.4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric2(2, 0, 4, 3, 1, 6), 0.4)

def test_find_pairs_hyper():
    # example: w=2, l=1, n=3
    pairs = itertools.product(range(3+1), range(3+1))
    pairs = itertools.filterfalse(lambda y: sum(y) != 3, pairs)
    pairs = itertools.filterfalse(lambda y: y[0] - y[1] < (2-1), pairs)
    expected_p = [(2, 1), (3, 0)]
    assert list(pairs)==expected_p
    
    # example: w=4, l=1, n=5
    pairs = itertools.product(range(5+1), range(5+1))
    pairs = itertools.filterfalse(lambda y: sum(y) != 5, pairs)
    pairs = itertools.filterfalse(lambda y: y[0] - y[1] < (4-1), pairs)
    expected_p = [(4, 1), (5, 0)]
    assert list(pairs)==expected_p
    
    
def test_diluted_margin_pvalue_hyper():
    # example 1: w=2, l=1, n=3, W=L=U=2
    t3 = 1*2/sp.misc.comb(4, 3)   # w=2, l=1, u=0
    t4 = 0                        # w=3, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_hypergeometric(2, 1, 2, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric2(2, 1, 2, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric3(2, 1, 2, 2), t3+t4)
    
    # example 1: w=4, l=1, n=5, W=5, L=U=2
    t3 = sp.misc.comb(5, 4)*2/sp.misc.comb(7, 5)   # w=4, l=1, u=0
    t4 = 1*1/sp.misc.comb(7, 5)                    # w=5, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_hypergeometric(4, 1, 5, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric2(4, 1, 5, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric3(4, 1, 5, 2), t3+t4)


### Run tests
if __name__ == "__main__": 
    test_find_pairs_trihyper()
    test_diluted_margin_pvalue_trihyper()
    test_find_pairs_hyper()
    test_diluted_margin_pvalue_hyper()