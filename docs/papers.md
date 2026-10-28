# Papers

## Conference paper

M. S. Samhan, J. Guerrero Orbe, N. Haghdadi, D. Sim, A. Bruce and B. Yildiz, "Reverse power
flow sign correction for meter-limited substation data," accepted for the Australasian
Universities Power Engineering Conference (AUPEC), November 2026.

## Journal article (in preparation)

A journal article on the current method, M9, is in preparation. This page will cite it once it
is published.

## Citing the software

If you use PyNRPF, please cite it using the metadata in
[`CITATION.cff`](https://github.com/mssamhan31/PyNRPF/blob/main/CITATION.cff), or the
archived release:

> Samhan, M. S., Bruce, A., & Yildiz, B. (2026). *PyNRPF: Detecting and rectifying wrong
> reverse power flow readings* [Software]. Zenodo. <https://doi.org/10.5281/zenodo.18764943>

/// details | Show details
    type: more

## About the conference paper

The paper identifies the wrong reverse power flow (RPF) sign that magnitude-only substation
meters produce in Australian distribution networks. It presents two earlier methods, a
deterministic rule (M7) and a machine-learning classifier (M8), which detect the error at
both the day level and the reading level. It also reports a case study in the Ausgrid
network on how correcting the sign changes minimum demand. The code and data that produced
it are archived, unchanged, under
[`publication/1_conference_paper/`](https://github.com/mssamhan31/PyNRPF/tree/main/publication/1_conference_paper),
and the software version it used is release 0.3.1 on Zenodo.

## About the journal article

The journal article covers how M9 works, its validation on reviewed real-world errors, and
the effect of sign correction on net-load forecasting. Its code and data are under
[`publication/2_journal_article/`](https://github.com/mssamhan31/PyNRPF/tree/main/publication/2_journal_article);
[Data](data.md), [Evaluation](evaluation.md) and [Reproduce the paper](reproduce.md) describe
them.

## Version DOIs

The software DOI above always resolves to the latest archived release; each release also has
its own version DOI on Zenodo.

///
