import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
from ballot_comparison import ballot_comparison_pvalue
from hypergeometric import trihypergeometric_optim
from sprt import ballot_polling_sprt
import matplotlib.pyplot as plt
import numpy.testing

def fisher_combined_pvalue(pvalues):
    """
    Find the p-value for Fisher's combined test statistic

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        p-value for Fisher's combined test statistic
    """
    if np.any(np.array(pvalues)==0):
        return 0
    obs = -2*np.sum(np.log(pvalues))
    return 1-scipy.stats.chi2.cdf(obs, df=2*len(pvalues))


def create_modulus(n1, n2, n_w2, n_l2, N1, V_wl, gamma):
    """
    The modulus of continuity for the Fisher's combined p-value.
    This function returns the modulus of continuity, as a function of
    the distance between two lambda values.
    
    n1 : int
        sample size in the ballot comparison stratum
    n2 : int
        sample size in the ballot polling stratum
    n_w2 : int
        votes for the reported winner in the ballot polling sample
    n_l2 : int
        votes for the reported loser in the ballot polling sample
    N1 : int
        total number of votes in the ballot comparison stratum
    V_wl : int
        margin (in votes) between w and l in the whole contest
    gamma : float
        gamma from the ballot comparison audit
    """
    Wn = n_w2; Ln = n_l2; Un = n2-n_w2-n_l2
    assert Wn>=0 and Ln>=0 and Un>=0
    
    return lambda delta: 2*Wn*np.log(1 + V_wl*delta) + 2*Ln*np.log(1 + 2*V_wl*delta) + \
            2*Un*np.log(1 + 3*V_wl*delta) + 2*n1*np.log(1 + V_wl*delta/(2*N1*gamma))


def maximize_fisher_combined_pvalue(N_w1, N_l1, N1, N_w2, N_l2, N2,
    pvalue_funs, stepsize=0.05, modulus=None, alpha=0.05, feasible_lambda_range=None):
    """
    Grid search to find the maximum P-value.
    
    Find the smallest Fisher's combined statistic for P-values obtained 
    by testing two null hypotheses at level alpha using data X=(X1, X2).

    Parameters
    ----------
    N_w1 : int
        votes for the reported winner in the ballot comparison stratum
    N_l1 : int
        votes for the reported loser in the ballot comparison stratum
    N1 : int
        total number of votes in the ballot comparison stratum
    N_w2 : int
        votes for the reported winner in the ballot polling stratum
    N_l2 : int
        votes for the reported loser in the ballot polling stratum
    N2 : int
        total number of votes in the ballot polling stratum
    pvalue_funs : array_like
        functions for computing p-values. The observed statistics/sample and known parameters should be
        plugged in already. The function should take the lambda allocation AS INPUT and output a p-value.
    stepsize : float
        size of the grid for searching over lambda. Default is 0.05
    modulus : function
        the modulus of continuity of the Fisher's combination function.
        This should be created using `create_modulus`.
        Optional (Default is None), but increases the precision of the grid search.
    alpha : float
        Risk limit. Default is 0.05.
    feasible_lambda_range : array-like
        lower and upper limits to search over lambda. 
        Optional, but a smaller interval will speed up the search.
    
    Returns
    -------
    dict with 
    
    max_pvalue: float
        maximum combined p-value
    min_chisq: float
        minimum value of Fisher's combined test statistic
    allocation lambda : float
        the parameter that minimizes the Fisher's combined statistic/maximizes the combined p-value
    refined : bool
        was the grid search refined after the first pass?
    stepsize : float
        the final grid step size used
    tol : float
        if refined is True, this is an upper bound on potential approximation error of min_chisq
    """	
    assert len(pvalue_funs)==2
    
    # find range of possible lambda
    if feasible_lambda_range is None:
        feasible_lambda_range = calculate_lambda_range(N_w1, N_l1, N1, N_w2, N_l2, N2)
    (lambda_lower, lambda_upper) = feasible_lambda_range

    test_lambdas = np.arange(lambda_lower, lambda_upper+stepsize, stepsize)
    if len(test_lambdas) < 5:
        stepsize = (lambda_upper + 1 - lambda_lower)/5
        test_lambdas = np.arange(lambda_lower, lambda_upper+stepsize, stepsize)
    fisher_pvalues = np.empty_like(test_lambdas)
    for i in range(len(test_lambdas)):
        pvalue1 = np.min([1, pvalue_funs[0](test_lambdas[i])])
        pvalue2 = np.min([1, pvalue_funs[1](1-test_lambdas[i])])
        fisher_pvalues[i] = fisher_combined_pvalue([pvalue1, pvalue2])
        
    pvalue = np.max(fisher_pvalues)
    alloc_lambda = test_lambdas[np.argmax(fisher_pvalues)]
    
    # If p-value is over the risk limit, then there's no need to refine the
    # maximization. We have a lower bound on the maximum.
    if pvalue > alpha or modulus is None:
        return {'max_pvalue' : pvalue,
                'min_chisq' : sp.stats.chi2.ppf(1 - pvalue, df=4),
                'allocation lambda' : alloc_lambda,
                'tol' : None,
                'stepsize' : stepsize,
                'refined' : False
                }
    
    # Use modulus of continuity for the Fisher combination function to check
    # how close this is to the true max
    fisher_fun_obs = scipy.stats.chi2.ppf(1-pvalue, df=4)
    fisher_fun_alpha = scipy.stats.chi2.ppf(1-alpha, df=4)
    dist = np.abs(fisher_fun_obs - fisher_fun_alpha)
    mod = modulus(stepsize)

    if mod <= dist:
        return {'max_pvalue' : pvalue,
                'min_chisq' : fisher_fun_obs,
                'allocation lambda' : alloc_lambda,
                'stepsize' : stepsize,
                'tol' : mod,
                'refined' : False
                }
    else:
        lambda_lower = alloc_lambda - 2*stepsize
        lambda_upper = alloc_lambda + 2*stepsize
        refined = maximize_fisher_combined_pvalue(N_w1, N_l1, N1, N_w2, N_l2, N2,
            pvalue_funs, stepsize=stepsize/10, modulus=modulus, alpha=alpha, 
            feasible_lambda_range=(lambda_lower, lambda_upper))
        refined['refined'] = True
        return refined


def plot_fisher_pvalues(N, overall_margin, pvalue_funs, alpha=None):
    """
    Plot the Fisher's combined p-value for varying error allocations 
    using data X=(X1, X2) 

    Parameters
    ----------
    N : array_like
        Array of stratum sizes
    overall_margin : int
        the difference in votes for reported winner and loser in the population
    pvalue_funs : array_like
        functions for computing p-values. The observed statistics/sample and known parameters should be plugged in already. The function should take the lambda allocation AS INPUT and output a p-value.
    alpha : float
        Optional, desired upper percentage point
    
    Returns
    -------
    dict with 
    
    float
        maximum combined p-value
    float
        minimum value of Fisher's combined test statistic
    float
        lambda, the parameter that minimizes the Fisher's combined statistic/maximizes the combined p-value
    """
    assert len(N)==2
    assert len(pvalue_funs)==2
        
    # find range of possible lambda
    (lambda_lower, lambda_upper) = calculate_lambda_range(N_w1, N_l1, N1, N_w2, N_l2, N2)

    fisher_pvalues = []
    cvr_pvalues = []
    nocvr_pvalues = []
    for lam in np.arange(lambda_lower, lambda_upper+1, 0.5):
        pvalue1 = np.min([1, pvalue_funs[0](lam)])
        pvalue2 = np.min([1, pvalue_funs[1](1-lam)])
        fisher_pvalues.append(fisher_combined_pvalue([pvalue1, pvalue2]))

    plt.scatter(np.arange(lambda_lower, lambda_upper+1, 0.5), fisher_pvalues, color='black')
    if alpha is not None:
        plt.axhline(y=alpha, linestyle='--', color='gray')
    plt.xlabel("Allocation of Allowable Error")
    plt.ylabel("Fisher Combined P-value")
    plt.show()
    
    
def simulate_fisher_combined_audit(N_w1, N_l1, N1, N_w2, N_l2, N2, n1, n2, alpha,
    reps=10000, verbose=False, plausible_lambda_range=None):
    """
    Simulate the Fisher method of combining a ballot comparison audit
    and ballot polling audit, assuming the reported results are correct.
    Return the fraction of simulations where the the audit successfully
    confirmed the election results.
    
    Parameters
    ----------
    N_w1 : int
        votes for the reported winner in the ballot comparison stratum
    N_l1 : int
        votes for the reported loser in the ballot comparison stratum
    N1 : int
        total number of votes in the ballot comparison stratum
    N_w2 : int
        votes for the reported winner in the ballot polling stratum
    N_l2 : int
        votes for the reported loser in the ballot polling stratum
    N2 : int
        total number of votes in the ballot polling stratum
    n1 : int
        sample size in the ballot comparison stratum
    n2 : int
        sample size in the ballot polling stratum
    alpha : float
        risk limit
    reps : int
        number of times to simulate the audit. Default 10,000
    verbose : bool
        Optional, print iteration number if True
    plausible_lambda_range : array-like
        lower and upper limits to search over lambda. Optional, but will speed up the search
    
    Returns
    -------
    float : fraction of simulations where the the audit successfully
    confirmed the election results
    """
    margin = (N_w1+N_w2)-(N_l1+N_l2)
    N1 = N_w1+N_l1
    N2 = N_w2+N_l2
    Vwl = (N_w1 + N_w2) - (N_l1 + N_l2)
    pop2 = [1]*N_w2 + [0]*N_l2 + [np.nan]*(N2 - N_w2 - N_l2)
    
    cvr_pvalue = lambda alloc: ballot_comparison_pvalue(n=n1, gamma=1.03905, \
                                   o1=0, u1=0, o2=0, u2=0,
                                   reported_margin=margin, N=N1,
                                   null_lambda=alloc)
    fisher_pvalues = np.zeros(reps)
    
    for i in range(reps):
        if verbose:
            print(i)
        sam = np.random.choice(pop2, n2, replace=False)
        nw2 = np.sum(sam == 1)
        nl2 = np.sum(sam == 0)
        mod = create_modulus(n1, n2, nw2, nl2, N1, Vwl, 1.03905)
        nocvr_pvalue = lambda alloc: \
            ballot_polling_sprt(sample=sam, popsize=N2, alpha=alpha,
                                Vw=N_w2, Vl=N_l2, \
                                null_margin=(N_w2-N_l2) - alloc*margin)['pvalue']
        fisher_pvalues[i] = maximize_fisher_combined_pvalue(N_w1, N_l1, 
                               N1, N_w2, N_l2, N2,
                               pvalue_funs=[cvr_pvalue, nocvr_pvalue],
                               modulus=mod,
                               plausible_lambda_range=plausible_lambda_range)['max_pvalue']
    return np.mean(fisher_pvalues <= alpha)


def calculate_lambda_range(N_w1, N_l1, N1, N_w2, N_l2, N2):
    V1 = N_w1 - N_l1
    V2 = N_w2 - N_l2
    V = V1+V2
    lb = np.min([2*N1/V, 1+2*N2/V,1-(N2+V2)/V])
    ub = np.max([-2*N1/V, 1-2*N2/V,  1+(N2-V2)/V])
    return (lb, ub)
    
    
def bound_fisher_fun(N_w1, N_l1, N1, N_w2, N_l2, N2,
                     pvalue_funs, plausible_lambda_range=None, stepsize=0.5):
        """
        DEPRECATED: Create piecewise constant upper and lower bounds for the Fisher's 
        combination function for varying error allocations

        Parameters
        ----------
        N_w1 : int
            votes for the reported winner in the ballot comparison stratum
        N_l1 : int
            votes for the reported loser in the ballot comparison stratum
        N1 : int
            total number of votes in the ballot comparison stratum
        N_w2 : int
            votes for the reported winner in the ballot polling stratum
        N_l2 : int
            votes for the reported loser in the ballot polling stratum
        N2 : int
            total number of votes in the ballot polling stratum
        pvalue_funs : array_like
            functions for computing p-values. The observed statistics/sample and known parameters should be plugged in already. 
            The function should take the lambda allocation AS INPUT and output a p-value.
        plausible_lambda_range : array-like
            lower and upper limits to search over lambda. Optional, but will speed up the search
        stepsize : float
            size of the mesh to calculate bounds; default 0.5
    
        Returns
        -------
        dict with 
    
        array-like
           sample_points : Fisher's combining function evaluated at the grid points
        array-like
           lower_bounds : piecewise constant lower bound on Fisher's combining function between the grid points
        array-like
           upper_bounds : piecewise constant upper bound on Fisher's combining function between the grid points
        array-like
           grid : grid of lambdas
        """
        if plausible_lambda_range is None:
            plausible_lambda_range = calculate_lambda_range(N_w1, N_l1, N1, N_w2, N_l2, N2)
        (lambda_lower, lambda_upper) = plausible_lambda_range
        
        cvr_pvalue = pvalue_funs[0]
        nocvr_pvalue = pvalue_funs[1]
        cvr_pvalues = []
        nocvr_pvalues = []
        
        for lam in np.arange(lambda_lower, lambda_upper+1, stepsize):
            pvalue1 = np.min([1, cvr_pvalue(lam)])
            pvalue2 = np.min([1, nocvr_pvalue(1-lam)])
            cvr_pvalues.append(pvalue1)
            nocvr_pvalues.append(pvalue2)
            
        lower_bounds = [fisher_combined_pvalue([cvr_pvalues[i+1], nocvr_pvalues[i]]) for i in range(len(cvr_pvalues)-1)]
        upper_bounds = [fisher_combined_pvalue([cvr_pvalues[i], nocvr_pvalues[i+1]]) for i in range(len(cvr_pvalues)-1)]
        sample_points = [fisher_combined_pvalue([cvr_pvalues[i], nocvr_pvalues[i]]) for i in range(len(cvr_pvalues))]
        
        return {'sample_points' : sample_points,
                'upper_bounds' : upper_bounds,
                'lower_bounds' : lower_bounds,
                'grid' : np.arange(lambda_lower, lambda_upper+1, stepsize)
                }


################################################################################
############################## Unit tests ######################################
################################################################################

def test_modulus1():
    N1 = 1000
    N2 = 100
    Vw1 = 550
    Vl1 = 450
    Vw2 = 60
    Vl2 = 40
    Vu2 = N2-Vw2-Vl2
    Vwl = (Vw1 + Vw2) - (Vl1 + Vl2)

    Nw1_null = N1/2
    n1 = 100
    o1 = o2 = u1 = u2 = 0
    gamma = 1.03

    nw2 = 6; Wn = nw2
    nl2 = 4; Ln = nl2
    n2 = 10; Un = n2 - nl2 - nw2

    def c(lam):
        return Vw2 - Vl2 - lam*Vwl

    def Nw_null(lam):
        return (N2 - Un + c(lam))/2

    def fisher_fun(lam):
        Nw_null_val = Nw_null(lam)
        c_val = c(lam)
        fisher_fun = -2*np.sum(np.log(Nw_null_val - np.arange(Wn))) +\
        -2*np.sum(np.log(Nw_null_val - c_val - np.arange(Ln))) +\
        -2*np.sum(np.log(N2 - 2*Nw_null_val + c_val - np.arange(Un))) +\
        2*np.sum(np.log(Vw2 - np.arange(Wn))) +\
        2*np.sum(np.log(Vl2 - np.arange(Ln))) +\
        2*np.sum(np.log(Vu2 - np.arange(Un))) +\
        -2*n1*np.log(1 - (1-lam)*Vwl/(2*N1*gamma)) + \
        2*o1*np.log(1 - 1/(2*gamma)) + \
        2*o2*np.log(1 - 1/(gamma)) + \
        2*u1*np.log(1 + 1/(2*gamma)) + \
        2*u2*np.log(1 + 1/(gamma))
        return fisher_fun
        
    mod = create_modulus(n1, n2, nw2, nl2, N1, Vwl, gamma)
    
    v1 = np.abs(fisher_fun(0.6 + 0.1) - fisher_fun(0.6))
    v2 = mod(0.1)
    np.testing.assert_array_less(v1, v2)
    v1 = np.abs(fisher_fun(0.2 + 0.01) - fisher_fun(0.2))
    v2 = mod(0.01)
    np.testing.assert_array_less(v1, v2)
    v1 = np.abs(fisher_fun(0.8 + 0.001) - fisher_fun(0.8))
    v2 = mod(0.001)
    np.testing.assert_array_less(v1, v2)
    
    
if __name__ == "__main__":
    test_modulus1()