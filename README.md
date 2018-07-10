[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-1.ipynb)

# A New Method for Stratified Risk-Limiting Audits

by Kellie Ottoboni, Philip B. Stark, Mark Lindeman, and Neal McBurnett

Risk-limiting audits (RLAs) offer a statistical guarantee that if a full manual tally would
show that the reported election outcome is wrong, an RLA has a known minimum chance of
leading to a full manual tally.
RLAs generally rely on random samples; risk calculations are simplest for random samples of individual ballots drawn with replacement from all validly cast ballots.
However, stratified sampling---partitioning the population of ballots into disjoint
strata, and sampling independently from the strata---may simplify logistics or make the audit more efficient.
For example, some Colorado counties (comprising 98.2\% of voters)
have new voting systems that allow auditors to check how the system interpreted each ballot; the rest do not.
%Before this work, the only approaches to combining information
%from all counties into a single RLA either required all counties to 
%use \emph{ballot-polling}, which is not as efficient as \emph{ballot-level comparison}, which only the newer systems support, or to use
%variable batch size comparison audits, which would require exporting data from the older systems in a way that the counties have not yet undertaken, and weighted random sampling, which Colorado's audit software does not support.
Previous approaches to combining information from all counties into a single RLA of a statewide contest would require counties with new voting systems to use a less efficient method than their equipment permits, or would require major procedural changes.
We provide a simpler, more efficient approach: 
stratify cast ballots into ballots cast in counties with newer systems and those cast in counties with legacy systems; 
sample individual ballots from those strata independently; 
apply a generalization of ballot-level comparison auditing in the first stratum and of ballot-polling auditing in the second to test the hypothesis that the ``overstatement error'' in each stratum exceeds a threshold; combine the stratum-level results using Fisher's nonparametric combination of tests; and find the maximum combined $P$-value over all partitions of the error.
The audit can stop when the maximum is less than the risk limit.
The method for combining information from different strata and for combining different audit strategies (ballot-polling and comparison) is new, and immediately applicable in Colorado.
We provide an open-source reference implementation of the method and exemplar calculations in Jupyter notebooks.


[Fisher's combination method illustration](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Ffisher_combined_pvalue.ipynb)

[Example Notebook 1](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-1.ipynb)

[Example Notebook 2](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-2.ipynb)