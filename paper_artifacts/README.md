# MediTriageAI Paper Artifacts

This directory contains automatically generated publication-quality artifacts for the MediTriageAI research paper.

## Directory Structure
- \diagrams/\: Pure Python/Matplotlib architecture and pipeline diagrams (PDF/SVG/PNG).
- \igures/\: Performance plots (ROC, PR Curves, Confusion Matrices, Learning Curves).
- \	ables/\: LaTeX and CSV formats of dataset statistics, model comparisons, and ablations.
- \	emplates/\: Reusable evaluation templates for pending experiments.
- \manifests/\: JSON manifests documenting generation provenance.

## Reproducibility
To regenerate all artifacts deterministically from available experiment outputs, run:

\\ash
python -m scripts.reproduce_paper
\
> **Note**: This script will gracefully handle missing experiment data by generating placeholder templates and explicit logs rather than fabricating scientific results.
