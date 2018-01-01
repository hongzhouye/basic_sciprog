PREFIX=exercises
PDFLATEX=pdflatex
PDFVIEWER=preview

$(PREFIX).pdf: $(PREFIX).tex
	mkdir -p out
	$(PDFLATEX) --output-directory=out $(PREFIX).tex
	$(PDFVIEWER) out/$(PREFIX).pdf

pdf: $(PREFIX).tex
	mkdir -p out
	$(PDFLATEX) --output-directory=out $(PREFIX).tex
	$(PDFLATEX) --output-directory=out $(PREFIX).tex
	$(PDFVIEWER) out/$(PREFIX).pdf

view: out/$(PREFIX).pdf
	$(PDFVIEWER) out/$(PREFIX).pdf

clean:
	rm -rf out
