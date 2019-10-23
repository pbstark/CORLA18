from __future__ import division, print_function
import math
import numpy as np
import scipy as sp
import json
import csv

class Assertion:
    """
    Objects and methods for assertions about elections.
    An _assertion_ is a statement of the form 
      "the average value of this assorter applied to the ballots is greater than 1/2"
    The _assorter_ maps votes to nonnegative numbers not exceeding `upper_bound`
    """
    
    def __init__(self, assorter = None):
        """
        The assorter is callable; the upper bound is a nonnegative real.
        """
        self.assorter = assorter
        
    def set_assorter(self, assorter):
        self.assorter = assorter
        
    def get_assorter(self):
        return self.assorter

    def assort(self, cvr):
        return self.assorter(cvr)
    
    def assorter_mean(self, cvr_list):
        """
        find the mean of the assorter values for a list of CVRs        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        double
        """
        return np.mean(map(self.assorter, cvr_list))        
        
    def assorter_sum(self, cvr_list):
        """
        find the sum of the assorter values for a list of CVRs        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        double
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
        return 2*(self.assorter_sum(cvr_list)-len(cvr_list)/2)
        
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
        
      
    @classmethod
    def make_plurality_assertions(self, winners, losers):
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
        a dict of Assertions
        
        """
        assertions = {}
        for winr in winners:
            winr_func = lambda c: CVR.as_vote(CVR.get_vote_from_votes(winr, c))
            for losr in losers:
                wl_pair = winr + ' v ' + losr                
                losr_func = lambda c: CVR.as_vote(CVR.get_vote_from_votes(losr, c))
                assertions[wl_pair] = Assertion(Assorter(winner=winr_func, loser=losr_func, \
                                      upper_bound = 1))
        return assertions
    
    @classmethod
    def make_supermajority_assertion(self, winner, losers, share_to_win):
        """
        Construct a set of assertions that imply that the winner got at least a fraction fraction_to_win
        of the valid votes.
        
        An equivalent condition is:
        
        (votes for winner)/(2*share_to_win) + (invalid votes)/2 > 1/2.
        
        Thus the correctness of a super-majority outcome can be checked with a single assertion.
        
        A CVR with a mark for more than one candidate in the contest is considered an invalid vote.
            
        Parameters:
        -----------
        winner : 
            identifiers of winning candidate
        losers : list
            list of identifiers of losing candidate(s)
        share_to_win : double
            fraction of the valid votes the winner must get to win        
        
        Returns:
        --------
        a dict containing one Assertion
        
        """
        assert share_to_win > 1/2, "share_to_win must be at least 1/2"
        assert share_to_win < 1, "share_to_win must be less than 1"
        assertions = {}
        wl_pair = winner + ' v all'
        cands = losers.copy()
        cands.append(winner)
        func = lambda c: CVR.as_vote(CVR.get_vote_from_votes(winner, c))/(2*share_to_win) \
                         if CVR.has_one_vote(cands, c) else 1/2   
        assertions[wl_pair] = Assertion(Assorter(assort = func, upper_bound = 1/(2*share_to_win)))
        return assertions

    @classmethod
    def make_assertions_from_json(self, candidates, json_assertions):
        """
        Construct a dict of Assertion objects from a RAIRE-style json representation 
        of a list of assertions for a given contest.

        Parameters:
        -----------
        candidates : 
            list of identifiers for all candidates in relevant contest.

        json_assertions:
            Assertions to be tested for the relevant contest.

        Returns:
        --------
        A dict of assertions for each assertion specified in 'json_assertions'.
        """        
        # @Vanessa, @Michelle
        # 
        assertions = {}
        for assrtn in json_assertions:
            winner = assrtn['Winner']
            loser = assrtn['Loser']

            # Is this a 'winner only' assertion
            if assrtn['Winner-Only'] == "true":
                # CVR is a vote for the winner only if it has the 
                # winner as its first preference
                winner_func = lambda v: 1 if CVR.get_vote_from_votes(winner, v) == 1 else 0

                # CVR is a vote for the loser if they appear and the 
                # winner does not, or they appear before the winner
                loser_func = lambda v : rcv_lfunc_wo(winner, loser, v)

                wl_pair = winner + ' v ' + loser
                assertions[wl_pair] = Assertion(Assorter(winner=winner_func,
                    loser=loser_func, upper_bound=1))

                continue

            # The assertion is of type 'irv elimination'. 
            # Context is that all candidates in 'eliminated' have been
            # eliminated and their votes distributed to later preferences
            eliminated = [e for e in assrtn['Already-Eliminated']]
            remaining = [c for c in candidates if not c in eliminated]

            winner_func = lambda v : rcv_votefor_cand(winner, remaining, v)        
            loser_func = lambda v : rcv_votefor_cand(loser, remaining, v)
        
            # Don't know what key should be for assertion. Made something up.
            wl_given = winner + ' v ' + loser + ' elim ' + ' '.join(eliminated)

            assertions[wl_given] = Assertion(Assorter(winner=winner_func,
                loser = loser_func, upper_bound = 1))

        return assertions
    
    @classmethod
    def make_all_assertions(self, contests):
        all_assertions = {}
        for c in contests:
            scf = contests[c]['choice_function']
            winrs = contests[c]['reported_winners']
            losrs = [cand for cand in contests[c]['candidates'] if cand not in winrs]
            if scf == 'plurality':
                all_assertions[c] = Assertion.make_plurality_assertions(winrs, losrs)
            elif scf == 'supermajority':
                all_assertions[c] = Assertion.make_supermajority_assertion(winrs[0], losrs, \
                                  contests[c]['share_to_win'])
            elif scf == 'IRV':
                # Assumption: contests[c]['assertions'] yields list assertions in JSON format.
                all_assertions[c] = Assertion.make_assertions_from_json(contests[c]['candidates'], \
                    contests[c]['assertions'])
            else:
                raise NotImplementedError("Social choice function " + scf + " is not supported")
        return all_assertions

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
        maps cvr into double
    
    upper_bound : double
        a priori upper bound on the value the assorter assigns to any CVR

    The basic method is assort, but the constructor can be called with (winner, loser)
    instead. In that case,
    
        assort = (winner - loser + 1)/2

    """
        
    def __init__(self, assort=None, winner=None, loser=None, upper_bound = 1):
        """
        Constructs an Assorter.
        
        If assort is defined and callable, it becomes the class instance of assort
        
        If assort is None but both winner and loser are defined and callable,
           assort is defined to be 1/2 if winner=loser; winner, otherwise
           
        
        Parameters:
        -----------
        assort : callable
            maps a CVR into [0, \infty)
        winner : callable
            maps a CVR into {0, 1}
        loser  : callable
            maps a CVR into {0, 1}
        """        
        self.winner = winner
        self.loser = loser
        self.upper_bound = upper_bound
        if assort is not None:
            assert callable(assort), "assort must be callable"
            self.assort = assort
        else:
            assert callable(winner), "winner must be callable if assort is None"
            assert callable(loser),  "loser must be callable if assort is None"
            self.assort = lambda cvr: (self.winner(cvr) - self.loser(cvr) + 1)/2 
    
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

    def set_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound
        
    def get_upper_bound(self):
        return self.upper_bound


class CVR:
    """
    Generic class for cast-vote records.
    
    Note that the CVR class DOES NOT IMPOSE VOTING RULES. For instance, the social choice
    function might consider a CVR that contains two votes in a contest to be an overvote.
    
    Class method get_votefor returns the vote for a given candidate if the candidate is a key in the CVR,
        or False if the candidate is not in the CVR. 
        This allows very flexible representation of votes, including ranked voting.
        
        For instance, in a plurality contest with four candidates, a vote for Alice (and only Alice)
        could be represented by any of the following:
            {"Alice": True}
            {"Alice": "marked"}
            {"Alice": 5}
            {"Alice": 1, "Bob": 0, "Candy": 0, "Dan": ""}
            {"Alice": True, "Bob": False}
            
        bool(vote_for("Alice"))==True iff the CVR contains a vote for Alice, and 
        int(bool(vote_for("Alice")))==1 if the CVR contains a vote for Alice, and 0 otherwise.
                
        Ranked votes also have simple representation, e.g.,
            {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''}
        Then int(vote_for("Alice")) is Alice's rank.
    
     
    Methods:
    --------
    
    get_votefor :  
         get_votefor(candidate, cvr) = returns the value in the votes dict for the key `candidate`, or
         False if no such key exists
    set_votes :  
         updates the votes dict; overrides previous votes and/or creates votes for additional candidates
    get_votes : returns complete votes dict
    """
    
    def __init__(self, votes = {}):
        self.votes = votes
        
    def get_votes(self):
        return self.votes
    
    def set_votes(self, votes):
        self.votes.update(votes)
            
    def get_votefor(self, candidate):
        return CVR.get_vote_from_votes(candidate, self.votes)
    
    @classmethod
    def as_vote(self, v):
        return int(bool(v))
    
    @classmethod
    def as_rank(self, v):
        return int(v)
    
    @classmethod
    def get_vote_from_votes(self, candidate, votes):
        """
        Returns the vote for a candidate if the CVR contains a vote for that candidate; 
        otherwise returns False
        
        Parameters:
        -----------
        candidate : 
            identifier for candidate
        
        votes : dict
            a dict where candidate identifiers are keys
        
        Returns:
        --------
        vote
        """
        return False if candidate not in votes else votes[candidate]
    
    @classmethod
    def has_one_vote(self, candidates, votes):
        """
        Is there exactly one vote among the candidates?
        
        Parameters:
        -----------
        candidates : list
            list of identifiers of candidates
        
        Returns:
        ----------
        True if there is exactly one vote among those candidates, where a vote means that the 
        value for that key casts as boolean True.
        """
        votes = np.sum([0 if c not in votes else bool(votes[c]) for c in candidates])
        return True if votes==1 else False

# utilities

def rcv_lfunc_wo(winner, loser, vote):
    """
    Check whether vote is a vote for the loser with respect to a 'winner only' 
    assertion between the given 'winner' and 'loser'.  

    Parameters:
    -----------
    winner : 
        identifier for winning candidate

    loser : 
        identifier for losing candidate

    vote : dict

    Returns:
    --------
    1 if the given vote is a vote for 'loser' and 0 otherwise
    """
    rank_winner = CVR.get_vote_from_votes(winner, vote)
    rank_loser = CVR.get_vote_from_votes(loser, vote)

    if not bool(rank_winner) and bool(rank_loser):
        return 1

    if bool(rank_winner) and bool(rank_loser) and rank_loser < rank_winner:
        return 1

    return 0 

def rcv_votefor_cand(cand, remaining, vote):
    """
    Check whether 'vote' is a vote for the given candidate in the context
    where only candidates in 'remaining' remain standing.

    Parameters:
    -----------
    cand : 
        identifier for candidate

    remaining : 
        list of identifiers of candidates still standing

    vote : dict

    Returns:
    --------
    1 if the given vote is a vote for 'cand' and 0 otherwise. Essentially,
    if you reduce the ballot down to only those canidates in 'remaining',
    and 'cand' is the first preference, we return 1 and 0 otherwise.
    """
    if not cand in remaining:
        return 0

    rank_cand = CVR.get_vote_from_votes(cand, vote)

    if not bool(rank_cand):
        return 0

    for altc in remaining:
        if altc == cand:
            continue

        rank_altc = CVR.get_vote_from_votes(altc, vote)

        if bool(rank_altc) and rank_altc < rank_cand:
            return 0

    return 1 

       
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
        assert contests[c]['choice_function'] in ['IRV','plurality','supermajority'], \
                  'unsupported choice function ' + contests[c]['choice_function'] + ' in ' + c + ' contest'
        assert contests[c]['n_winners'] <= len(contests[c]['candidates']), \
            'fewer candidates than winners in ' + c + ' contest'
        assert len(contests[c]['reported_winners']) == contests[c]['n_winners'], \
            'number of reported winners does not equal n_winners in ' + c + ' contest'
        for w in contests[c]['reported_winners']:
            assert w in contests[c]['candidates'], \
                'reported winner ' + w + ' is not a candidate in ' + c + 'contest'
        if contests[c]['choice_function'] in ['IRV','supermajority']:
            assert contests[c]['n_winners'] == 1, \
                contests[c]['choice_function'] + ' can have only 1 winner in ' + c + ' contest'
        if contests[c]['choice_function'] == 'supermajority':
            assert contests[c]['share_to_win'] >= 0.5, \
                'super-majority contest requires winning at least 50% of votes in ' + c + ' contest'

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

# Unit tests

def test_supermajority_assorter():
    losers = ["Bob","Candy"]
    share_to_win = 2/3
    assn = Assertion.make_supermajority_assertion("Alice", losers, share_to_win)

    votes = {"Alice": 1}
    assert assn['Alice v all'].assorter.assort(votes) == 3/4, "wrong value for vote for winner"
    
    votes = {"Bob": True}
    assert assn['Alice v all'].assorter.assort(votes) == 0, "wrong value for vote for loser"
    
    votes = {"Dan": True}
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Dan"

    votes = {"Alice": True, "Bob": True}
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Alice & Bob"

    votes = {"Alice": False, "Bob": True, "Candy": True}
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Bob & Candy"
   

def test_rcv_lfunc_wo():
    votes = {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''}
    assert rcv_lfunc_wo("Bob", "Alice", votes) == 1
    assert rcv_lfunc_wo("Alice", "Candy", votes) == 0
    assert rcv_lfunc_wo("Dan", "Candy", votes) == 1

def test_rcv_votefor_cand():
    votes = {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": '', "Ross" : 4, "Aaron" : 5}
    remaining = ["Bob","Dan","Aaron","Candy"]
    assert rcv_votefor_cand("Candy", remaining, votes) == 0 
    assert rcv_votefor_cand("Alice", remaining, votes) == 0 
    assert rcv_votefor_cand("Bob", remaining, votes) == 1 
    assert rcv_votefor_cand("Aaron", remaining, votes) == 0

    remaining = ["Dan","Aaron","Candy"]
    assert rcv_votefor_cand("Candy", remaining, votes) == 1 
    assert rcv_votefor_cand("Alice", remaining, votes) == 0 
    assert rcv_votefor_cand("Bob", remaining, votes) == 0 
    assert rcv_votefor_cand("Aaron", remaining, votes) == 0
 

if __name__ == "__main__":
    test_supermajority_assorter()
    test_rcv_lfunc_wo()
    test_rcv_votefor_cand()    
