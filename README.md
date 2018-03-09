[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-1.ipynb)

# Next Steps for the Colorado Risk-Limiting Audit (CORLA) Program

by Mark Lindeman, Neal McBurnett, Kellie Ottoboni, and Philip B. Stark

Colorado conducted risk-limiting tabulation audits (RLAs) across the state in 2017,
including both ballot-level comparison audits and ballot-polling audits.
Those audits only covered contests restricted to a single county;
methods to efficiently audit contests that cross county boundaries
and combine ballot polling and ballot-level comparisons have not been available.

Colorado's current audit software (RLATool) needs to be improved to audit
these contests that cross county lines and to audit small contests efficiently.

This paper addresses these needs. 
It presents extremely simple but inefficient methods, more efficient methods
that combine ballot polling and ballot-level comparisons using stratified samples,
and methods that combine ballot-level comparison and
variable-size batch comparison audits in a way that does not require stratified
sampling.

We conclude with some recommendations, and illustrate our recommended method
using examples that compare them to existing approaches.
Exemplar open-source code and interactive Jupyter notebooks are provided
that implement the methods and allow further exploration.

[Example Notebook 1](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-1.ipynb)
[Example Notebook 2](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-2.ipynb)