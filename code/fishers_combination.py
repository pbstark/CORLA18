import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
from ballot_comparison import ballot_comparison_pvalue
from hypergeometric import trihypergeometric_optim
from sprt import ballot_polling_sprt
import matplotlib.pyplot as plt


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


def maximize_fisher_combined_pvalue(N, overall_margin, pvalue_funs, precise=True, plausible_lambda_range=None):
    """
    Find the smallest Fisher's combined statistic for p-values obtained 
    by testing two null hypotheses at level alpha using data X=(X1, X2) 

    Parameters
    ----------
    N : array_like
        Array of stratum sizes
    overall_margin : int
        the difference in votes for reported winner and loser in the population
    pvalue_funs : array_like
        functions for computing p-values. The observed statistics/sample and known parameters should be
        plugged in already. The function should take the lambda allocation AS INPUT and output a p-value.
    precise : bool
        Optional, should we refine the maximum found by minimize_scalar? Default is True
    plausible_lambda_range : array-like
        lower and upper limits to search over lambda. Optional, but will speed up the search
    
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
    if plausible_lambda_range is None:
        lambda_upper = int(np.min([2*N[0]/overall_margin, 1+2*N[1]/overall_margin]))+1
        lambda_lower = int(np.max([-2*N[0]/overall_margin, 1-2*N[1]/overall_margin]))
    else:
        lambda_upper = plausible_lambda_range[1]
        lambda_lower = plausible_lambda_range[0]

    fisher_pvalues = []
    cvr_pvalues = []
    test_lambdas = np.arange(lambda_lower, lambda_upper+1, 0.5)
    for lam in test_lambdas:
        pvalue1 = np.min([1, pvalue_funs[0](lam)])
        if pvalue1 < 0.01:
            fisher_pvalues.append(0)
        else:
            pvalue2 = np.min([1, pvalue_funs[1](1-lam)])
            fisher_pvalues.append(fisher_combined_pvalue([pvalue1, pvalue2]))
        
    pvalue = np.max(fisher_pvalues)
    alloc_lambda = test_lambdas[np.argmax(fisher_pvalues)]
    
    # go back and make sure that this is actually the maximizer within a window
    
    if precise is True:
        fisher_pvalues = []
        test_lambdas = np.arange(alloc_lambda-0.5, alloc_lambda+0.5, 0.1)
        for lam in test_lambdas:
            pvalue1 = np.min([1, pvalue_funs[0](lam)])
            pvalue2 = np.min([1, pvalue_funs[1](1-lam)])
            fisher_pvalues.append(fisher_combined_pvalue([pvalue1, pvalue2]))
            
        if np.max(fisher_pvalues) > pvalue:
            pvalue = np.max(fisher_pvalues)
            alloc_lambda = test_lambdas[np.argmax(fisher_pvalues)]
    
    return {'max_pvalue' : pvalue,
            'min_chisq' : sp.stats.chi2.ppf(1 - pvalue, df=4),
            'allocation lambda' : alloc_lambda
            }


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
#    (lambda_lower, lambda_upper) = calculate_lambda_range(N_w1, N_l1, N1, N_w2, N_l2, N2)
    lambda_upper = int(np.min([2*N[0]/overall_margin, 1+2*N[1]/overall_margin]))+1
    lambda_lower = int(np.max([-2*N[0]/overall_margin, 1-2*N[1]/overall_margin]))

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
        #nocvr_pvalue = lambda alloc: trihypergeometric_optim(sample=sam, 
        #                                popsize=N2, 
        #                                null_margin=(N_w2-N_l2) - alloc*margin)
        nocvr_pvalue = lambda alloc: \
            ballot_polling_sprt(sample=sam, popsize=N2, alpha=alpha,
                                Vw=N_w2, Vl=N_l2, \
                                null_margin=(N_w2-N_l2) - alloc*margin)['pvalue']
        fisher_pvalues[i] = maximize_fisher_combined_pvalue(N=(N1, N2),
                               overall_margin=margin, 
                               pvalue_funs=[cvr_pvalue, nocvr_pvalue],
                               precise=True,
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
        Create piecewise constant upper and lower bounds for the Fisher's 
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