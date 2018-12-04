from __future__ import division
import math
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats
import scipy.optimize

def ballot_polling_sprt(sample, popsize, alpha, Vw, Vl, 
                        null_margin=0, number_invalid=None):
    """
    Wald's SPRT for the difference in true number of votes for the winner, Nw, and the loser, Nl:
    
    H_0: Nw = Nl + null_margin
    H_1: Nw = Vw, Nl = Vl
    
    The type II error rate for the test, usually denoted beta, is set to 0%: if the data do not
    support rejecting the null, there is a full hand count.
    
    Because beta=0, the reciprocal of the likelihood ratio is a conservative p-value.
    
    Parameters
    ----------
    sample : array-like
        audit sample. Elements should equal 1 (ballots for w), 0 (ballots for l), or np.nan (the rest)
    popsize : int
        number of ballots cast in the stratum
    alpha : float
        desired type 1 error rate
    Vw : int
        number of votes for w in the stratum, under the alternative hypothesis
    Vl : int
        total number of votes for l in the stratum, under the alternative hypothesis
    null_margin : int
        vote margin between w and l under the null hypothesis; optional
        (default 0)
    number_invalid : int
        number of invalid ballots, undervoted ballots, or ballots for other candidates in 
        the stratum, if known; optional (default None)
        
    Returns
    -------
    dict
    """
    
    # Set parameters
    upper = 1/alpha
    n = len(sample)
    assert n <= popsize, "Sample size greater than population size"
    
    sample = np.array(sample)    
    Wn = np.sum(sample == 1)
    Ln = np.sum(sample == 0)
    Un = n - Wn - Ln
    
    upper_Nw_limit = popsize - Un - Ln
    lower_Nw_limit = np.max([Wn, Ln+null_margin, 0])  # if the null hypothesis is true
    
    if Wn > upper_Nw_limit or (lower_Nw_limit + Ln + Un) > popsize:
        return {'decision' : 'Null is impossible, given the sample',
                'upper_threshold' : upper,
                'LR' : np.inf,
                'pvalue' : 0,
                'sample_proportion' : (Wn/n, Ln/n, Un/n),
                'Nu_used' : None,
                'Nw_used' : None
                }
    
    decision = "None"

    # Set up likelihood for null and alternative hypotheses
    Vw = int(Vw)
    Vl = int(Vl)
    Vu = int(popsize - Vw - Vl)
    assert Vw >= Wn and Vl >= Ln and Vu >= Un, "Alternative hypothesis isn't consistent with the sample"
    
    alt_logLR = np.sum(np.log(Vw - np.arange(Wn))) + \
                np.sum(np.log(Vl - np.arange(Ln))) + \
                np.sum(np.log(Vu - np.arange(Un)))
        
    null_logLR = lambda Nw: (Wn > 0)*np.sum(np.log(Nw - np.arange(Wn))) + \
                (Ln > 0)*np.sum(np.log(Nw - null_margin - np.arange(Ln))) + \
                (Un > 0)*np.sum(np.log(popsize - 2*Nw + null_margin - np.arange(Un)))
    

    # This is just for testing the code. In practice, number_invalid is unknown.
    if number_invalid is not None:
        assert isinstance(number_invalid, int)
        assert number_invalid < popsize
        nuisance_param = (popsize - number_invalid + null_margin)/2
        if nuisance_param < lower_Nw_limit or nuisance_param > upper_Nw_limit:
            return {'decision' : 'Number invalid is incompatible with the null and the data',
                    'upper_threshold' : upper,
                    'LR' : np.inf,
                    'pvalue' : 0,
                    'sample_proportion' : (Wn/n, Ln/n, Un/n),
                    'Nu_used' : number_invalid,
                    'Nw_used' : nuisance_param
                    }
        
        res = alt_logLR - null_logLR(nuisance_param)
        LR = np.exp(res)

    else:   # this is the typical case, Nu unknown                
        LR_derivative = lambda Nw: (Wn > 0)*np.sum([1/(Nw - i) for i in range(Wn)]) + \
                    (Ln > 0)*np.sum([1/(Nw - null_margin - i) for i in range(Ln)]) - \
                    (Un > 0)*2*np.sum([1/(popsize - 2*Nw + null_margin - i) for i in range(Un)])

        # Sometimes the upper_Nw_limit is too extreme, causing illegal 0s.
        # Check and change the limit when that occurs.
        if np.isinf(null_logLR(upper_Nw_limit)):
            upper_Nw_limit -= 1

        # Check if the maximum occurs at an endpoint
        if LR_derivative(upper_Nw_limit)*LR_derivative(lower_Nw_limit) > 0:  # deriv has no sign change
            nuisance_param = upper_Nw_limit if null_logLR(upper_Nw_limit)>=null_logLR(lower_Nw_limit) else lower_Nw_limit
        # Otherwise, find the (unique) root of the derivative of the log likelihood ratio
        else:
            nuisance_param = sp.optimize.brentq(LR_derivative, lower_Nw_limit, upper_Nw_limit)
#            nuisance_param = np.floor(nuisance_param) if null_logLR(np.floor(nuisance_param))>=null_logLR(np.ceil(nuisance_param)) else np.ceil(nuisance_param)
        number_invalid = popsize - 2*nuisance_param + null_margin  # N - Nw - Nl
        
        logLR = alt_logLR - null_logLR(nuisance_param)
        LR = np.exp(logLR)

    if logLR >= np.log(upper):
        # reject the null and stop
        decision = 1
            
    return {'decision' : decision,
            'upper_threshold' : upper,
            'LR' : LR,
            'pvalue' : min(1, 1/LR),
            'sample_proportion' : (Wn/n, Ln/n, Un/n),
            'Nu_used' : number_invalid,
            'Nw_used' : nuisance_param
            }


###################### Unit tests ############################

def test_sprt_functionality():
    trials = np.zeros(100)
    trials[0:50] = 1
    res = ballot_polling_sprt(trials, popsize=1000, alpha=0.05, Vw=500, Vl=450)
    assert res['decision']=='None'
    assert res['lower_threshold']==0.0
    assert res['upper_threshold']==20.0
    assert res['pvalue']>0.05
    assert res['LR']<1
    assert res['sample_proportion']==(0.5, 0.5, 0)
    
    trials[50:60] = 1
    res = ballot_polling_sprt(trials, popsize=1000, alpha=0.05, Vw=600, Vl=400)
    assert res['decision']=='None'
    assert res['lower_threshold']==0.0
    assert res['upper_threshold']==20.0
    assert res['pvalue']>0.05
    assert res['LR']>1
    assert res['sample_proportion']==(0.6, 0.4, 0)
    
    trials = np.zeros(250)
    trials[0:110] = 1
    trials[110:150] = np.nan
    res = ballot_polling_sprt(trials, popsize=1000, alpha=0.05, Vw=500, Vl=450)
    assert res['decision']=='None'
    assert res['lower_threshold']==0.0
    assert res['upper_threshold']==20.0
    assert res['pvalue']>0.05
    assert res['LR']<1
    assert res['sample_proportion']==(0.44, 0.4, 0.16)
            
    trials = np.zeros(100)
    trials[0:40] = 1        
    res = ballot_polling_sprt(trials, popsize=1000, alpha=0.05, Vw=500, Vl=450)
    assert res['decision']=='None'
    assert res['pvalue']==1
    assert res['LR']<1
    assert res['sample_proportion']==(0.4, 0.6, 0)


def test_sprt_analytic_example():
    sample = [0, 0, 1, 1]
    population = [0]*5 + [1]*5
    popsize = len(population)
    res = ballot_polling_sprt(sample, popsize, alpha=0.05, Vw=5, Vl=4, number_invalid=0)
    np.testing.assert_almost_equal(res['LR'], 0.6)
    np.testing.assert_almost_equal(res['Nu_used'], 0)
    res2 = ballot_polling_sprt(sample, popsize, alpha=0.05, Vw=5, Vl=4)
    np.testing.assert_almost_equal(res2['LR'], 0.6, decimal=2)
    np.testing.assert_almost_equal(res2['Nu_used'], 0)
    
    sample = [0, 1, 1, 1]
    res = ballot_polling_sprt(sample, popsize, alpha=0.05, Vw=6, Vl=4, number_invalid=0)
    np.testing.assert_almost_equal(res['LR'], 1.6)
    res2 = ballot_polling_sprt(sample, popsize, alpha=0.05, Vw=6, Vl=4)
    np.testing.assert_almost_equal(res2['LR'], 1.6, decimal=2)
    np.testing.assert_almost_equal(res2['Nu_used'], 0, decimal=2)

    sample = [0, 1, 1]
    res = ballot_polling_sprt(sample, popsize, alpha=0.05, Vw=6, Vl=4, number_invalid=0)
    np.testing.assert_almost_equal(res['LR'], 1.2)
    res2 = ballot_polling_sprt(sample, popsize, alpha=0.05, Vw=6, Vl=4)
    np.testing.assert_almost_equal(res2['LR'], 1.2, decimal=2)
    np.testing.assert_almost_equal(res2['Nu_used'], 0, decimal=2)



if __name__ == 'main':
    test_sprt_functionality()
    test_sprt_analytic_example()