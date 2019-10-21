import svgling
from svgling.figure import Caption, SideBySide, RowByRow

# Convert a tree in list form into the same tree in tuple form suitable for
# svgling.
def treeListToTuple(t):
    # If t is an empty list, we shouldn't have got this far.
    if not t:
        print("Error: empty list in tree drawing")
    # Leaf.  Return the name of the candidate and the assertion we've excluded it with.
    if len(t) == 1:     
        return((t[0][0],t[0][1]+str(t[0][2]))) 
    # Otherwise recurse.
    else:
        tList = []
        for branch in t[1]:
            tList.append(treeListToTuple(branch))
        return ((t[0],)+tuple(tList))

def printSixTrees(elimTrees):
    print("Built "+str(len(elimTrees))+" trees.")
    print("Warning: hardcoded to print 6 trees!")
    Caption(RowByRow(RowByRow(RowByRow(elimTrees[0],elimTrees[1]),RowByRow(elimTrees[2],elimTrees[3])),RowByRow(elimTrees[4],elimTrees[5])   ), "Whole trees excluded.")

def parseAssertions(auditfile):
    apparentWinner = auditfile["Audits"][0]["Winner"]
    print("Apparent winner: "+apparentWinner)
    apparentNonWinners=auditfile["Audits"][0]["Eliminated"]
    print("Apparently eliminated: ")
    print(apparentNonWinners)
    assertions = auditfile["Audits"][0]["Assertions"]

    # WOLosers is a set of tuples - the first element of the tuple is the loser,
    # the second element is a list of all the candidates it loses wrt.
    WOLosers = []
    # IRVElims is also a set of tuples - the first element is the candidate,
    # the second is the set of candidates already eliminated.
    # An IRVElim assertion states that the candidate can't be the next
    # eliminated when the already-eliminated candidates are exactly the set
    # in the second element of the tuple.
    IRVElims = []
    for a in assertions:
        if a["Winner-Only"]=="true":
            l = a["Loser"]
            w = a["Winner"]
            # if we haven't already encountered this loser, add a new element to WOLosers.
            # if we have, add a new winner to this loser's set.
            losers = [ll for ll,_ in WOLosers]
            if l not in losers:
                #if l not in [losers[0] for losers in WOLosers]
                WOLosers.append((l,set(w)))
            else:
                for losers in WOLosers:
                    if l == losers[0]:
                        losers[1].add(w)
                    
        if a["Winner-Only"]=="false":
            l = a["Winner"]
            IRVElims.append((l,set(a["Already-Eliminated"])  ))
    return(apparentWinner, apparentLoser, WOLosers, IRVElims)
