TEX_DOCS := CO18.tex stratumRisk.tex subcollections.tex pbsBib.bib

CO18.pdf: $(TEX_DOCS)
	latexmk -pdf CO18
