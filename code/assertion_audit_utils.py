from __future__ import division, print_function
import math
import numpy as np
import scipy as sp
import json
import csv

    
class Assertion:
    def __init__(self, assorter = None):
        self.assorter = assorter
        
    def set_assorter(self, assorter):
        self.assorter = assorter
        
    def get_assorter(self):
        return self.assorter

    def assort(self, cvr):
        return self.assorter(cvr)
    
    def category_sum(self, cvr_list):
        """
        find the sum of the category values for a list of CVRs        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        total : double
        """
        return np.sum(map(self.assorter, cvr_list))        
        
    def margin_votes(self, cvr_list):
        """
        find the margin in votes for a list of CVRs
        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        margin : double
        """
        return 2*(self.category_sum(cvr_list)-len(cvr_list)/2)
        
    def margin_fraction(self, cvr_list):
        """
        find the diluted margin (as a fraction) for a list of CVRs
        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        margin : double
        """
        return self.margin_votes(cvr_list)/len(cvr_list)
    
    def overstatement(self, mvr, cvr):
        """
        Find the overstatement error (in votes) in a CVR compared to the human 
        reading of the ballot
        
        Parameters:
        -----------
        mvr : Cvr
            the manual interpretation of voter intent
        cvr : Cvr
            the machine-reported cast vote record
            
        Returns:
        --------
        overstatement error
        """
        return 2*(self.assorter(mvr)-self.assorter(cvr))
    
    def overstatement_count(self, mvr_list, cvr_list):
        """
        Count the discrepancies between human reading of a collection of ballots and the
        CVRs for those ballots
        
        Parameters:
        -----------
        mvr_list : list
            list of manually determined CVRs
        cvr_list : list
            list of the machine-reported CVRs
            
        Returns:
        --------
        tuple : number of ballots with overstatements of -2, -1, 1, and 2.
        
        """
        assert len(mvr_list) == len(cvr_list), "number of mvrs differs from number of cvrs"
        discrepancies = np.array(map(self.overstatement, mvr_list, cvr_list))
        o2 = np.sum(discrepancies == 2)
        o1 = np.sum(discrepancies == 1)
        u1 = np.sum(discrepancies == -1)
        u2 = np.sum(discrepancies == -2)
        return [o2, o1, u1, u2]
        
        
    class Assorter:
        """
        Class for generic Assorter.
        
        Class parameters:
        -----------------
        winner : callable
            maps a CVR into the value 1 if the CVR represents a vote for the winner        
        loser  : callable
            maps a CVR into the value 1 if the CVR represents a vote for the winner
        
        assort : callable
            maps cvr into {0, 1/2, 1}:
               0 if loser==1 and winner==0
               1 if loser==0 and winner==1
               1/2 otherwise

        The basic method is assort, but the constructor can be called with (winner, loser)
        instead. In that case, assort is defined as follows:
        
            If the they both map the same CVR to the same value, assort = 1/2. 
            Otherwise, assort=0 if loser==1 and assort=1 if winner==1.
        """
            
        def __init__(self, assort=None, winner=None, loser=None):
            """
            Constructs an Assorter.
            
            If assort is defined and callable, is becomes the class instance of assort
            
            If assort is None but both winner and loser are defined and callable,
               assort=1/2 if winner=loser; assort=winner, otherwise
            
            Parameters:
            -----------
            assort : callable
                maps a CVR into {0, 1/2, 1}
            winner : callable
                maps a CVR into {0, 1}
            loser  : callable
                maps a CVR into {0, 1}
            """            
            self.winner = winner
            self.loser = loser
            if assort is not None:
                assert callable(assort), "assort must be callable"
                self.assort = assort
            else:
                assert callable(winner), "assort is None so winner must be callable"
                assert callable(loser), "assort is None so loser must be callable"
                self.assort = lambda cvr: self.winner(cvr) \
                              if self.winner(cvr) != self.loser(cvr) else 1/2 
        
        def set_winner(self, winner):
            self.winner = winner

        def get_winner(self):
            return(self.winner)

        def set_loser(self, loser):
            self.loser = loser

        def get_loser(self):
            return(self.loser)
        
        def set_assort(self):
            self.assort = assort

        def get_assort(self):
            return(self.assort)
      

    def make_plurality_assertions(winners, losers):
        """
        Construct a set of assertions that imply that the winner(s) got more votes than the loser(s).
        
        The assertions are that every winner beat every loser: there are
              len(winners)*len(losers) 
        pairwise assertions in all.
        
        Parameters:
        -----------
        winners : list
            list of identifiers of winning candidate(s)
        losers : list
            list of identifiers of losing candidate(s)
        
        Returns:
        --------
        a set of Assertion objects
        
        """
        assertions = set()
        for winr in winners:
            for losr in losers:
                winr_func = lambda c: CVR.is_votefor(winr, c)
                losr_func = lambda c: CVR.is_votefor(losr, c)
                assertions.add(Assertion(Assorter(winner=winr_func, loser=losr_func)))
        return assertions
                
class CVR:
    """
    Generic class for cast-vote records.
    
    Class method for determining whether a CVR shows a vote for a particular candidate
    
    Methods:
    --------
    
    is_votefor : maps (cvr, candidate) into {False, True}. 
         is_votefor(cvr, candidate) = True iff the cvr shows a vote for candidate
         if None, the constructor checks whether candidate in cvr
    
    """
    
    def __init__(self, is_votefor = None):
        if is_votefor is not None:
            assert callable(is_votefor), "is_votefor must be callable"                
            self.is_votefor = is_votefor
        else:
            self.is_votefor = lambda cvr, cand: cand in cvr
        
    def get_is_votefor:
        return self.is_votefor
    
    def set_is_votefor:
        self.is_votefor = is_votefor
    


# utilities
            
def check_audit_parameters(gamma, error_rates, contests):
    """
    Check whether the audit parameters are valid; complain if not.
    
    Parameters:
    ---------
    gamma : double
        value of gamma for the Kaplan-Markov method of Lindeman and Stark (2012)
        
    error_rates : dict
        expected rates of overstatement and understatement errors
        
    contests : dict of dicts 
        contest-specific information for the audit
        
    Returns:
    --------
    """
    assert gamma >=1, 'gamma must be at least 1'
    for r in ['o1_rate','o2_rate','u1_rate','u2_rate']:
        assert error_rates[r] >= 0, 'expected error rates must be nonnegative'
    for c in contests.keys():
        assert contests[c]['risk_limit'] > 0, 'risk limit must be nonnegative in ' + c + ' contest'
        assert contests[c]['risk_limit'] < 1, 'risk limit must be less than 1 in ' + c + ' contest'
        assert contests[c]['choice_function'] in ['IRV','plurality','super-majority'], \
                  'unsupported choice function ' + contests[c]['choice_function'] + ' in ' + c + ' contest'
        assert contests[c]['n_winners'] <= len(contests[c]['candidates']), 'fewer candidates than winners in ' + c + ' contest'
        assert len(contests[c]['reported_winners']) == contests[c]['n_winners'], 'number of reported winners does not equal n_winners in ' + c + ' contest'
        for w in contests[c]['reported_winners']:
            assert w in contests[c]['candidates'], 'reported winner ' + w + ' is not a candidate in ' + c + 'contest'
        if contests[c]['choice_function'] in ['IRV','super-majority']:
            assert contests[c]['n_winners'] == 1, contests[c]['choice_function'] + ' can have only 1 winner in ' + c + ' contest'
        if contests[c]['choice_function'] == 'super-majority':
            assert contests[c]['super_majority'] >= 0.5, 'super-majority contest requires winning at least 50% of votes in ' + c + ' contest'

def write_audit_parameters(log_file, seed, replacement, gamma, N_ballots, error_rates, contests):
    """
    Write audit parameters to log_file as a json structure
    
    Parameters:
    ---------
    log_file : string
        filename to write to
        
    seed : string
        seed for the PRNG for sampling ballots
        
    gamma : double
        value of gamma for the Kaplan-Markov method of Lindeman and Stark (2012)
        
    error_rates : dict
        expected rates of overstatement and understatement errors
        
    contests : dict of dicts 
        contest-specific information for the audit
        
    Returns:
    --------
    """
    out = {"seed" : seed,
           "replacement" : replacement,
           "gamma" : gamma,
           "N_ballots" : N_ballots,
           "error_rates" : error_rates,
           "contests" : contests
          }
    with open(log_file, 'w') as f:
        json.dump(out, f)

def write_ballots_sampled(ballot_file, ballots):
    """
    Write the identifiers of the sampled ballots to a file.
    
    Parameters:
    ----------
    
    ballot_file : string
        filename for output
        
    ballots : list of lists
        ballot number, batch identifier, location of ballot within batch, number of times ballot was selected
    
    Returns:
    --------
    """
    
    with open(ballot_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ballot", "batch", "ballot_in_batch", "times_sampled"])
        for row in ballots:
            writer.writerow(row)

if __name__ == "__main__":
    pass
    