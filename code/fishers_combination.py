import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
from ballot_comparison import ballot_comparison_pvalue
from hypergeometric import trihypergeometric_optim
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


def maximize_fisher_combined_pvalue(N, overall_margin, pvalue_funs):
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
    lambda_upper = int(np.min([2*N[0]/overall_margin, 1+2*N[1]/overall_margin]))
    lambda_lower = int(np.max([-2*N[0]/overall_margin, 1-2*N[1]/overall_margin]))

    fisher_fun = lambda lam: -1*fisher_combined_pvalue([pvalue_funs[0](lam), pvalue_funs[1](1-lam)])
    res = sp.optimize.minimize_scalar(fisher_fun, bounds=(lambda_lower, lambda_upper), method='bounded')
    alloc_lambda = res['x']
    pvalue = -1*res['fun']
    
    # go back and make sure that this is actually the maximizer within a window
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
    lambda_upper = int(np.min([2*N[0]/overall_margin, 1+2*N[1]/overall_margin]))
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
