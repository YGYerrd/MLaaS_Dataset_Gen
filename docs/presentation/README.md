# Project presentation materials

Aligned with:

- [`../reference/MATH3004 Project Proposal.pdf`](../reference/MATH3004%20Project%20Proposal.pdf)
- [`../reference/ICSOC_2026_paper_28.pdf`](../reference/ICSOC_2026_paper_28.pdf)

## Primary format: PowerPoint (editable)

| File | Purpose |
| --- | --- |
| **`MLaaS_presentation.pptx`** | Native PowerPoint deck — edit text, shapes, charts, and layout in PowerPoint |
| [`build_presentation.py`](./build_presentation.py) | Regenerates the `.pptx` from Python (`python-pptx`) |
| [`talking-points.md`](./talking-points.md) | Speaker script with section refs, citations, Q&A |
| [`ICSOC_Table1_MLaaS_datasets.docx`](./ICSOC_Table1_MLaaS_datasets.docx) | ICSOC 2026 Table 1 (Word table, Harvey-ball ratings) |
| [`slides.md`](./slides.md) | Legacy Marp source (optional); not required if you use the `.pptx` |

### Build or refresh the deck

```powershell
cd docs\presentation
pip install -r requirements-presentation.txt
python build_presentation.py
```

Output: `docs/presentation/MLaaS_presentation.pptx` (17 slides — incl. **Key contribution** & **Limitations**)

### Editing in PowerPoint

- Every title, bullet, table, and card is a **separate shape** you can click and edit.
- Add figures from the ICSOC paper (Fig. 1, 4, 6–8) via **Insert → Pictures**.
- Replace the workflow slide text with **Insert → SmartArt** or copy Fig. 2 from the PDF.
- Theme colours match the generator script (blue `#2563eb`, dark `#1e3a8a`).

### When to re-run the script

Re-run `build_presentation.py` only if you change slide **content** in the Python file and want to reset the deck. For normal tweaks (wording, fonts, images), edit `MLaaS_presentation.pptx` directly and avoid overwriting your changes.

## Customization checklist

- [ ] Open `MLaaS_presentation.pptx` in PowerPoint and adjust branding/images
- [ ] Add ICSOC paper figures (composition scenario, ASQ charts)
- [ ] Rehearse with `talking-points.md` (~14 min full deck)
- [ ] Confirm proposal vs paper vs repo boundary (see talking-points Appendix D)
