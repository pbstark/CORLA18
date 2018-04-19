from __future__ import division, print_function
import math
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats


def ballot_comparison_pvalue(n, gamma, o1, u1, o2, u2, reported_margin, N, null_lambda=1):
    """
    Compute the p-value for a ballot comparison audit using Kaplan-Markov
    
    Parameters
    ----------
    n : int
        sample size
    gamma : float
        value > 1 to inflate the error bound, to avoid requiring full hand count for a single 2-vote overstatement
    o1 : int
        number of ballots that overstate any 
        margin by one vote but no margin by two votes
    u1 : int
        number of ballots that understate any margin by 
        exactly one vote, and every margin by at least one vote
    o2 : int
        number of ballots that overstate any margin by two votes
    u2 : int
        number of ballots that understate every margin by two votes
    reported_margin : float
        the smallest reported margin *in votes* between a winning
        and losing candidate for the contest as a whole, including any other strata
    N : int
        number of votes cast in the stratum
    null_lambda : float
        fraction of the overall margin (in votes) to test for in the stratum. If the overall margin is reported_margin,
        test that the overstatement in this stratum does not exceed null_lambda*reported_margin

    Returns
    -------
    pvalue
    """
    U_s = 2*N/reported_margin
    log_pvalue = n*np.log(1 - null_lambda/(gamma*U_s)) - \
                    o1*np.log(1 - 1/(2*gamma)) - \
                    o2*np.log(1 - 1/gamma) - \
                    u1*np.log(1 + 1/(2*gamma)) - \
                    u2*np.log(1 + 1/gamma)
    pvalue = np.exp(log_pvalue)
    return np.min([pvalue, 1])


def findNmin_ballot_comparison(alpha, gamma, o1, u1, o2, u2,
                                reported_margin, N, null_lambda=1):

    """
    Compute the smallest sample size for which a ballot comparison 
    audit, using Kaplan-Markov, with the given statistics could stop
    
    Parameters
    ----------
    alpha : float
        risk limit
    gamma : float
        value > 1 to inflate the error bound, to avoid requiring full hand count for a single 2-vote overstatement
    o1 : int
        number of ballots that overstate any 
        margin by one vote but no margin by two votes
    u1 : int
        number of ballots that understate any margin by 
        exactly one vote, and every margin by at least one vote
    o2 : int
        number of ballots that overstate any margin by two votes
    u2 : int
        number of ballots that understate every margin by two votes
    reported_margin : float
        the smallest reported margin *in votes* between a winning
        and losing candidate in the contest as a whole, including any other strata
    N : int
        number of votes cast in the stratum 
    null_lambda : float
        fraction of the overall margin (in votes) to test for in the stratum. If the overall margin is reported_margin,
        test that the overstatement in this stratum does not exceed null_lambda*reported_margin
        
    Returns
    -------
    n
    """
    U_s = 2*N/reported_margin
    val = -gamma*U_s/null_lambda * (np.log(alpha) +
                o1*np.log(1 - 1/(2*gamma)) + \
                o2*np.log(1 - 1/gamma) + \
                u1*np.log(1 + 1/(2*gamma)) + \
                u2*np.log(1 + 1/gamma) )
    val2 = o1+o2+u1+u2
    return np.max([int(val)+1, val2])


def findNmin_ballot_comparison_rates(alpha, gamma, r1, s1, r2, s2,
                                reported_margin, N, null_lambda=1):

    """
    Compute the smallest sample size for which a ballot comparison 
    audit, using Kaplan-Markov, with the given statistics could stop
    
    Parameters
    ----------
    alpha : float
        risk limit
    gamma : float
        value > 1 to inflate the error bound, to avoid requiring full hand count for a single 2-vote overstatement
    r1 : int
        hypothesized rate of ballots that overstate any 
        margin by one vote but no margin by two votes
    s1 : int
        hypothesizedrate of ballots that understate any margin by 
        exactly one vote, and every margin by at least one vote
    r2 : int
        hypothesizedrate of ballots that overstate any margin by two votes
    s2 : int
        hypothesizedrate of ballots that understate every margin by two votes
    reported_margin : float
        the smallest reported margin *in votes* between a winning
        and losing candidate in the contest as a whole, including any other strata
    N : int
        number of votes cast in the stratum
    null_lambda : float
        fraction of the overall margin (in votes) to test for in the stratum. If the overall margin is reported_margin,
        test that the overstatement in this stratum does not exceed null_lambda*reported_margin
        
    Returns
    -------
    n
    """
    U_s = 2*N/reported_margin
    denom = (np.log(1 - null_lambda/(U_s*gamma)) -
                r1*np.log(1 - 1/(2*gamma))- \
                r2*np.log(1 - 1/gamma) - \
                s1*np.log(1 + 1/(2*gamma)) - \
                s2*np.log(1 + 1/gamma) )
    return np.ceil(np.log(alpha)/denom) if denom < 0 else np.nan



# unit tests from "A Gentle Introduction..."
def gentle_intro_tests():
    np.testing.assert_array_less(ballot_comparison_pvalue(80, 1.03905, 0,1,0,0,5,100), 0.1)
    np.testing.assert_array_less(ballot_comparison_pvalue(96, 1.03905, 0,0,0,0,5,100), 0.1)
    np.testing.assert_equal(findNmin_ballot_comparison(0.1, 1.03905, 0,1,0,0,5,100), 80)
    np.testing.assert_equal(findNmin_ballot_comparison(0.1, 1.03905, 0,0,0,0,5,100), 96)

# unit tests from pbstark/S157F17/audit.ipynb
def stat157_tests():
    np.testing.assert_equal(ballot_comparison_pvalue(n=200, gamma=1.03905, o1=1, u1=0, o2=0, u2=0, 
                            reported_margin=(354040 - 337589), N=354040+337589+33234),
                            0.21438135077031845)
    np.testing.assert_equal(findNmin_ballot_comparison_rates(alpha=0.05, gamma=1.03905, 
                            r1=.001, r2=0, s1=.001, s2=0,
                            reported_margin=5, N=100),
                            125)
    assert math.isnan(findNmin_ballot_comparison_rates(alpha=0.05, gamma=1.03905, 
                                        r1=.05, r2=0, s1=0, s2=0,
                                        reported_margin=5, N=100))
    

if __name__ == "__main__":
    gentle_intro_tests()
    stat157_tests()