import svgling
from svgling.figure import Caption

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

