"""Run python3 hypergeometric.py from the command line to run the unit tests"""

from __future__ import division
import math
import numpy as np
import numpy.random
import scipy as sp
from scipy.special import comb, gammaln as gamln
import scipy.stats
from scipy.optimize import minimize_scalar
import itertools

### Tri-hypergeometric distribution tests

def trihypergeometric_logpmf(w, l, n, N_w, N_l, N):
    return gamln(N_w+1) - gamln(N_w-w+1) - gamln(w+1) \
            + gamln(N_l+1) - gamln(N_l-l+1) - gamln(l+1) \
            + gamln(N-N_w-N_l+1) - gamln(N-N_w-N_l-n+w+l+1) - gamln(n-w-l+1) \
            - gamln(N+1) + gamln(N-n+1) + gamln(n+1)


def trihypergeometric_pmf(w, l, n, N_w, N_l, N):
    return np.exp(trihypergeometric_logpmf(w, l, n, N_w, N_l, N))


def diluted_margin_trihypergeometric_gamma(w, l, n, N_w, N_l, N):
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
    exact : bool, optional
        If exact is False, then floating point precision is used, 
        otherwise exact long integer is computed.
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
    return sum(map(lambda p: trihypergeometric_pmf(p[0], p[1], n, N_w, N_l, N), pairs))


def trihypergeometric_optim(sample, popsize, null_margin):
    '''
    Wrapper function for p-value calculations using the tri-hypergeometric distribution.
    This function maximizes the p-value over all possible values of the nuisance parameter,
    the number of votes for the reported winner in the population.
    
    The maximization is done on the continuous approximation to the p-value, using gamma functions.
    The maximum here is an upper bound on the true maximum, which must occur at an integer value
    of the nuisance parameter N_w. Here, the maximum can occur at a non-integer value.
    
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
    optim_fun = lambda N_w: -1*diluted_margin_trihypergeometric_gamma(w, l, n, N_w, N_w-null_margin, popsize)
    # conditions are that N_w+N_l = 2*upper - c < N-u, N_l = upper-c > l, N_w = upper > w
    upper_Nw = int((popsize-u+null_margin)/2)
    lower_Nw = int(np.max([w, null_margin]))
    
    res = minimize_scalar(optim_fun, 
                       bracket = [lower_Nw, upper_Nw],
                       method = 'brent')
    if res['x'] > upper_Nw:
        pvalue = -1*optim_fun(upper_Nw)
    elif res['x'] < lower_Nw:
        pvalue = -1*optim_fun(lower_Nw)
    else:
        pvalue = -1*res['fun']
    return pvalue



def diluted_margin_trihypergeometric(w, l, n, N_w, N_l, N, exact=True):
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
    exact : bool, optional
        If exact is False, then floating point precision is used, 
        otherwise exact long integer is computed.
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
    return sum(map(lambda p: comb(N_w, p[0], exact=exact)*\
                         comb(N_l, p[1], exact=exact)*\
                         comb(N_u, n-p[0]-p[1], exact=exact),\
                         pairs))/comb(N, n, exact=exact)


def diluted_margin_trihypergeometric2(w, l, n, N_w, N_l, N, exact=True):
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
    exact : bool, optional
        If exact is False, then floating point precision is used, 
        otherwise exact long integer is computed.
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
                tmp += comb(N_l, ll, exact=exact)*comb(N_u, n-ww-ll, exact=exact)
        pvalue += tmp * comb(N_w, ww, exact=exact)
    return pvalue/comb(N, n, exact=exact)


def trihypergeometric_optim_bruteforce(sample, popsize, null_margin, exact=True):
    '''
    Wrapper function for p-value calculations using the tri-hypergeometric distribution.
    This function maximizes the p-value over all possible values of the nuisance parameter,
    the number of votes for the reported winner in the population.
    
    The maximization is done by brute force, computing the tri-hypergeometric p-value at all
    possible integer values of the nuisance parameter N_w. This can be very slow.
    
    Parameters
    ----------
    sample : array-like
        sample of ballots. Values must be 0 (votes for l), 1 (votes for w), and np.nan (other votes).
    popsize : int
        total number of ballots in the population
    null_margin : int
        largest difference in *number* of votes between the reported winner and reported loser,
        N_w - N_l, under the null hypothesis
    exact : bool, optional
        If exact is False, then floating point precision is used, 
        otherwise exact long integer is computed.
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
    optim_fun = lambda N_w: diluted_margin_trihypergeometric(w, l, n, N_w, N_w-null_margin, popsize,
                                exact=exact)
    # conditions are that N_w+N_l = 2*upper - c < N-u, N_l = upper-c > l, N_w = upper > w
    upper_Nw = int((popsize-u+null_margin)/2)
    lower_Nw = int(np.max([w, null_margin]))
    return np.max(list(map(optim_fun, range(lower_Nw, upper_Nw+1))))


def gen_sample(w, n):
    """
    Helper function for `simulate_ballot_polling_power`
    """
    if w>0:
        sample = np.array([0]*0 + [1]*w + [np.nan]*(n-w))
    else:
        sample = np.array([0]*-w + [1]*0 + [np.nan]*(n+w))
    return sample
    

def simulate_ballot_polling_power(N_w, N_l, N, null_margin, n, alpha, reps=10000,
    stepsize=5, seed=987654321, verbose=True):
    """
    Simulate the power of the trihypergeometric ballot polling audit.
    This simulation assumes that the reported vote totals are true and
    draws `reps` samples of size n from the population, then computes
    the proportion of samples for which the audit could stop.
    
    Parameters
    ----------
    N_w : int
        total number of *reported* votes for w in the population
    N_l : int
        total number of *reported* votes for l in the population
    N : int
        total number of ballots in the population
    null_margin : int
        largest difference in *number* of votes between the reported winner and reported 
        loser, N_w - N_l, under the null hypothesis
    n : int
        number of ballots in the sample
    alpha : float
        risk limit
    reps : int
        number of simulation runs. Default is 10000
    stepsize : int
        when searching for the threshold margin, what step size to use? Default is 5
    seed : int
        random seed value for the pseudorandom number generator. Default is 987654321
    verbose : bool
        print (margin, pvalue) pairs? Default is True
    """
    np.random.seed(seed)
    
    # step 1: find diluted margin for which we'd reject
    # the p-value depends only on the margin, not the values of w and l
    if verbose:
        print("Step 1: find diluted margin for which the p-value <= alpha")
    w = int(n*N_w/N)
    sample = np.array([0]*0 + [1]*w + [np.nan]*(n-w))
    pvalue_mar = trihypergeometric_optim(sample, N, null_margin)
    if verbose:
        print(w, pvalue_mar)
    if pvalue_mar <= alpha:
        while pvalue_mar <= alpha and w<=n and w>=-n:
            w = w-stepsize
            sample = gen_sample(w, n)
            pvalue_mar = trihypergeometric_optim(sample, N, null_margin)
            if verbose:
                print(w, pvalue_mar)
        while pvalue_mar > alpha and w<=n and w>=-n:
            w = w+1
            sample = gen_sample(w, n)
            pvalue_mar = trihypergeometric_optim(sample, N, null_margin)
            if verbose:
                print(w, pvalue_mar)
        threshold = w
    else:
        while pvalue_mar > alpha and w<=n and w>=-n:
            w = w+stepsize
            sample = gen_sample(w, n)
            pvalue_mar = trihypergeometric_optim(sample, N, null_margin)
            if verbose:
                print(w, pvalue_mar)
        while pvalue_mar <= alpha and w<=n and w>=-n:
            w = w-1
            sample = gen_sample(w, n)
            pvalue_mar = trihypergeometric_optim(sample, N, null_margin)
            if verbose:
                print(w, pvalue_mar)
        threshold = w+1
    print("The critical value of the test is ", threshold)
            
    # step 2: over many samples, compute diluted margin
    population = np.array([0]*int(N_l) + [1]*int(N_w) + [np.nan]*(N-N_w-N_l))
    rejects = 0
    for r in range(reps):
        sample = np.random.choice(population, size=n)
        obs_mar = np.sum(sample==1) - np.sum(sample==0)
        if obs_mar >= threshold:
            rejects += 1

    # step 3: what fraction of these are >= the threshold?
    return rejects/reps


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
    t1 = 2*1*1/comb(6, 3) # w=1, l=0, u=2
    t2 = 1*1*2/comb(6, 3) # w=2, l=0, u=1
    t3 = 1*2*1/comb(6, 3) # w=2, l=1, u=0
    t4 = 0                        # w=3, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric(2, 1, 3, 2, 2, 6), t1+t2+t3+t4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric2(2, 1, 3, 2, 2, 6), t1+t2+t3+t4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric_gamma(2, 1, 3, 2, 2, 6), t1+t2+t3+t4)
    
    sample = np.array([1]*2 + [0]*1)
    pvalue1 = trihypergeometric_optim(sample, popsize=6, null_margin=0)
    pvalue2 = trihypergeometric_optim_bruteforce(sample, popsize=6, null_margin=0)
    np.testing.assert_array_less(t1+t2+t3+t4, pvalue1)
    np.testing.assert_array_less(t1+t2+t3+t4, pvalue2)
    np.testing.assert_almost_equal(pvalue1, pvalue2, decimal=1)
    
    # example 2: w=4, l=1, n=5, W=5, L=U=2
    t1 = comb(5, 3)*1*1/comb(9, 5) # w=3, l=0, u=2
    t2 = comb(5, 4)*1*2/comb(9, 5) # w=4, l=0, u=1
    t3 = comb(5, 4)*2*1/comb(9, 5) # w=4, l=1, u=0
    t4 = 1*1*1/comb(9, 5)                  # w=5, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric(4, 1, 5, 5, 2, 9), t1+t2+t3+t4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric2(4, 1, 5, 5, 2, 9), t1+t2+t3+t4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric_gamma(4, 1, 5, 5, 2, 9), t1+t2+t3+t4)
    
    sample = np.array([1]*4 + [0]*1)
    pvalue1 = trihypergeometric_optim(sample, popsize=9, null_margin=3)
    pvalue2 = trihypergeometric_optim_bruteforce(sample, popsize=9, null_margin=3)
    np.testing.assert_array_less(t1+t2+t3+t4, pvalue1)
    np.testing.assert_array_less(t1+t2+t3+t4, pvalue2)
    np.testing.assert_almost_equal(pvalue1, pvalue2, decimal=1)
    
    # example 3: w=2, l=0, n=4, W=3, L=1, N=6. Result should be 0.4
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric(2, 0, 4, 3, 1, 6), 0.4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric2(2, 0, 4, 3, 1, 6), 0.4)
    np.testing.assert_almost_equal(diluted_margin_trihypergeometric_gamma(2, 0, 4, 3, 1, 6), 0.4)
    
    sample = np.array([1]*2 + [np.nan]*2)
    pvalue1 = trihypergeometric_optim(sample, popsize=6, null_margin=2)
    pvalue2 = trihypergeometric_optim_bruteforce(sample, popsize=6, null_margin=2)
    np.testing.assert_almost_equal(0.4, pvalue1, decimal=1)
    np.testing.assert_almost_equal(0.4, pvalue2, decimal=1)
    np.testing.assert_almost_equal(pvalue1, pvalue2, decimal=1)
    
    
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
    t3 = 1*2/comb(4, 3)   # w=2, l=1, u=0
    t4 = 0                        # w=3, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_hypergeometric(2, 1, 2, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric2(2, 1, 2, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric3(2, 1, 2, 2), t3+t4)
    
    # example 1: w=4, l=1, n=5, W=5, L=U=2
    t3 = comb(5, 4)*2/comb(7, 5)   # w=4, l=1, u=0
    t4 = 1*1/comb(7, 5)                    # w=5, l=0, u=0
    np.testing.assert_almost_equal(diluted_margin_hypergeometric(4, 1, 5, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric2(4, 1, 5, 2), t3+t4)
    np.testing.assert_almost_equal(diluted_margin_hypergeometric3(4, 1, 5, 2), t3+t4)


### Run tests
if __name__ == "__main__": 
    test_find_pairs_trihyper()
    test_diluted_margin_pvalue_trihyper()
    test_find_pairs_hyper()
    test_diluted_margin_pvalue_hyper()