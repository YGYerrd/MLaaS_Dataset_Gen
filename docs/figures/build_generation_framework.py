"""Build the new-generation framework as editable SVG and vector PDF.

Run with Python + reportlab. Outputs live beside this script, except the PDF
under outputs/documents/. Render the PDF with Poppler to produce the PNG preview.

Scope: the current independent, raw_source_family_v4, recipe-only workflow.
The diagram is a logical dataflow, not the literal CLI execution chronology:
hf-manifest plans contracts/recipes; run-manifest freezes durable families and
executes raw-row selection, preparation, training, measurement and persistence.

Implementation sources (relative to repository root):
  mlaas_data_generator/cli/manifest/hf_manifest_builder.py
  mlaas_data_generator/cli/manifest/preparation.py
  mlaas_data_generator/clients/{policies,training,partitioning,quantities}.py
  mlaas_data_generator/data/tokenization_selection.py
  mlaas_data_generator/services/runner.py
  mlaas_data_generator/client_reconstruction.py
  mlaas_data_generator/execution_provenance.py
  mlaas_data_generator/storage/schema.sql
Interpretation and downstream scope:
  docs/client-reconstruction.md; docs/generation-policies.md
  docs/metric-evaluation-protocol.md; docs/recipe-composition.md
  docs/journal/reuse-protocols.md

Figure caption: Source model/dataset contracts define compatible families with
pinned dependencies, shared initial weights, catalogue evaluation rows and a
separate audit reservation. Versioned recipes control client data and training
heterogeneity. Each client is independently trained from the common initial
state and measured on the frozen catalogue subset. Typed service attributes,
effective execution evidence and reconstruction recipes populate a SQLite
benchmark catalogue linked to content-addressed artifacts. Downstream analyses
consume catalogue evidence; supported composition selections can reconstruct
members, aggregate compatible parameters and measure the result on saved audit
rows. Dashed connectors carry shared dependencies and audit data. Recipe-only
retention stores the shared initial state and per-client recipes; replay repeats
training and its functional fidelity is evaluated separately.
"""

from __future__ import annotations

from html import escape
import json
from math import atan2, cos, sin
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF = ROOT / "outputs" / "documents" / "dataset-generator-new-mode-framework.pdf"
W, H, SCALE = 2160, 1970, 0.45
INK = "#172D43"
TEXT = "#344B60"
MUTED = "#566D80"
LINE = "#CAD5DE"
BLUE = "#2E5D97"
TEAL = "#007C83"
PURPLE = "#7251A1"
ORANGE = "#A55C14"
BG = "#F7F9FB"
svg: list[str] = []
checks: list[dict] = []


def colour(value):
    return HexColor(value)


def rect(x, y, width, height, fill="#FFFFFF", stroke=LINE, radius=14, lw=1.5):
    c.setFillColor(colour(fill))
    c.setStrokeColor(colour(stroke or fill))
    c.setLineWidth(lw)
    c.roundRect(x, H-y-height, width, height, radius, stroke=bool(stroke), fill=1)
    svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{lw}"/>')


def text(x, y, value, size=22, fill=TEXT, bold=False, max_width=None, anchor="start"):
    font = "Helvetica-Bold" if bold else "Helvetica"
    width = stringWidth(value, font, size)
    if max_width is not None:
        checks.append({"text": value, "width": round(width, 2), "limit": max_width})
        if width > max_width:
            raise ValueError(f"Text too wide ({width:.1f} > {max_width}): {value}")
    if not 0 <= y <= H or not 0 <= x <= W:
        raise ValueError(f"Text outside page: {value}")
    c.setFillColor(colour(fill))
    c.setFont(font, size)
    draw = c.drawRightString if anchor == "end" else c.drawCentredString if anchor == "middle" else c.drawString
    draw(x, H-y, value)
    svg.append(f'<text x="{x}" y="{y}" fill="{fill}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(value)}</text>')


def lines(x, y, values, size=22, leading=29, **kwargs):
    for i, value in enumerate(values):
        text(x, y+i*leading, value, size=size, **kwargs)


def path(points, stroke=BLUE, width=3, dashed=False, arrow=True):
    c.setStrokeColor(colour(stroke))
    c.setLineWidth(width)
    c.setDash([9, 7] if dashed else [])
    p = c.beginPath()
    p.moveTo(points[0][0], H-points[0][1])
    for x, y in points[1:]:
        p.lineTo(x, H-y)
    c.drawPath(p)
    c.setDash([])
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}/>')
    if arrow:
        end, prev = points[-1], points[-2]
        angle = atan2(end[1]-prev[1], end[0]-prev[0])
        vertices = [end, (end[0]-15*cos(angle)+6*sin(angle), end[1]-15*sin(angle)-6*cos(angle)),
                    (end[0]-15*cos(angle)-6*sin(angle), end[1]-15*sin(angle)+6*cos(angle))]
        c.setFillColor(colour(stroke))
        p = c.beginPath()
        p.moveTo(vertices[0][0], H-vertices[0][1])
        for x, y in vertices[1:]:
            p.lineTo(x, H-y)
        p.close()
        c.drawPath(p, stroke=0, fill=1)
        svg.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in vertices)+f'" fill="{stroke}"/>')


def panel(x, y, width, height, number, title, subtitle, accent=BLUE):
    rect(x, y, width, height)
    rect(x, y, width, 7, fill=accent, stroke=None, radius=2)
    rect(x+24, y+24, 43, 43, fill=accent, stroke=None, radius=10)
    text(x+45.5, y+54, number, size=25, fill="#FFFFFF", bold=True, anchor="middle")
    text(x+82, y+49, title, size=27, bold=True, fill=INK, max_width=width-106)
    text(x+82, y+78, subtitle, size=18, fill=MUTED, max_width=width-106)


def group(x, y, label, body, width=550, accent=BLUE, size=22):
    text(x, y, label, size=19, bold=True, fill=accent, max_width=width)
    lines(x, y+30, body, size=size, max_width=width)


def tag(x, y, width, value, fill="#EEF3F8", accent=BLUE):
    rect(x, y, width, 34, fill=fill, stroke=None, radius=7)
    text(x+12, y+23, value, size=17, bold=True, fill=accent, max_width=width-24)


def build():
    global c
    PDF.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(PDF), pagesize=(W*SCALE, H*SCALE), pageCompression=1)
    c.scale(SCALE, SCALE)
    c.setTitle("Dataset generator framework - independent recipe-only generation")
    c.setAuthor("MLaaS")
    c.setSubject("Eight-stage framework grounded in the local dataset generator implementation")
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
    svg.append('<title id="title">Dataset generator: independent recipe-only generation</title>')
    svg.append('<desc id="desc">Eight stages: source catalogue, family preparation, controlled recipes, independent client training and evaluation, metadata and QoS, provenance and reconstruction, benchmark database, and downstream tasks. A separate audit-data route bypasses catalogue selection.</desc>')
    rect(0, 0, W, H, fill=BG, stroke=None, radius=0)
    text(80, 64, "DATASET GENERATOR", size=20, bold=True, fill=BLUE)
    text(80, 112, "Framework for independent service generation", size=43, bold=True, fill=INK)
    tag(1560, 43, 520, "NEW MODE  /  INDEPENDENT + RECIPE-ONLY", fill="#E4F2F0", accent=TEAL)
    text(2080, 111, "One shared initial state. Many controlled service variants.", size=20, fill=MUTED, anchor="end")

    # Stages 1-3: conceptual preparation and recipe planning.
    panel(80, 160, 600, 430, "01", "Source model / dataset catalogue", "Select compatible model-dataset-task contracts")
    group(105, 277, "MODEL SOURCES", ["Hugging Face models and local neural baselines", "Architecture, task head and parameter schema"])
    group(105, 369, "DATASET SOURCES", ["Dataset/configuration and source revisions", "Official training and separate evaluation splits"])
    group(105, 461, "CONTRACT PREFLIGHT", ["Task, modality, labels and split compatibility", "Record excluded or unsupported candidates"])
    tag(105, 541, 549, "Output: compatible source contracts + preflight report")

    panel(780, 160, 600, 430, "02", "Family preparation", "Pin and freeze shared dependencies", accent=TEAL)
    group(805, 277, "SEMANTIC FAMILY CONTRACT", ["Pinned sources; task, label and preprocessing rules", "Shared fitted transforms and parameter contract"], accent=TEAL)
    group(805, 369, "COMMON INITIAL STATE", ["Initialise the model / task head once; retain w0", "Every client resets to this same frozen state"], accent=TEAL)
    group(805, 461, "FROZEN DATA RESERVATIONS", ["Full training source; common catalogue subset", "Separate saved audit rows for downstream checks"], accent=TEAL)
    tag(805, 541, 550, "Output: family ID + contract + w0 + reservation", fill="#E4F2F0", accent=TEAL)

    panel(1480, 160, 600, 430, "03", "Controlled service recipes", "Versioned policies with seeded coverage")
    group(1505, 277, "CLIENT DATA VARIATION", ["Quantity, sampling seed and distribution intensity", "IID / Dirichlet / shard on supported sampling axes"])
    group(1505, 369, "TRAINING VARIATION", ["Optimizer, learning rate, batch size and epochs", "Accumulation, weight decay, warmup; feasibility"])
    group(1505, 461, "CATALOGUE-INFORMED POLICIES, WHERE ELIGIBLE", ["70% supported / 20% challenging / 10% stress", "Frozen historical evidence; explicit fallback policies"], size=21)
    tag(1505, 541, 550, "Output: planned manifest + policy hashes + coverage")
    path([(680, 361), (780, 361)])
    path([(1380, 361), (1480, 361)])

    # Direct dependency routes are distinct from the primary stage sequence.
    path([(960, 590), (960, 622), (420, 622), (420, 690)], stroke=TEAL, dashed=True)
    text(480, 613, "shared state + training source + catalogue rows", size=18, fill=TEAL)
    path([(1280, 590), (1280, 651), (2120, 651), (2120, 1750)], stroke=TEAL, dashed=True, arrow=False)
    text(1475, 641, "saved audit rows: reserved for downstream validation", size=18, fill=TEAL)
    path([(1770, 590), (1770, 614), (2050, 614), (2050, 690)])

    # Stage 4: explicit per-client independence and selection-before-preparation.
    panel(80, 690, 2000, 330, "04", "Heterogeneous client training and evaluation", "Repeat across compatible families; clients are independent experiments")
    rect(105, 792, 545, 202, fill="#F3F7FC", stroke=None)
    group(125, 825, "SELECT RAW ROWS, THEN PREPARE", ["Sample without replacement within each client", "Allow overlap across clients; record realised counts", "Prepare selected rows; replace rejected candidates", "Reuse shared preparation while preserving row order"], width=505, size=20)
    path([(650, 888), (720, 888)])
    rect(720, 792, 650, 202, fill="#ECF5F4", stroke=None)
    text(742, 825, "RESET w0 + OPTIMIZER / RNG STATE FOR EACH CLIENT", size=19, fill=TEAL, bold=True, max_width=606)
    for index, label in enumerate(["Client 1", "Client 2", "Client N"]):
        xx = 742 + index*207
        rect(xx, 844, 186, 78, fill="#FFFFFF", stroke="#A9D1CE", radius=9)
        text(xx+93, 873, label, size=21, bold=True, fill=INK, anchor="middle")
        text(xx+93, 903, "w0 + data + recipe", size=17, fill=TEAL, anchor="middle")
    text(742, 950, "Train separately; capture actual settings and updates", size=21, max_width=606)
    text(742, 979, "No client-to-client model transfer or generation aggregation", size=19, fill=MUTED, max_width=606)
    path([(1370, 888), (1440, 888)])
    rect(1440, 792, 615, 202, fill="#F3F7FC", stroke=None)
    group(1462, 825, "COMMON CATALOGUE EVALUATION", ["Evaluate every client on the same frozen subset", "Task-specific quality, timing and resource measures", "Trust / explanation protocols where supported", "Capture completion, failures and evidence status"], width=571, size=21)
    path([(1985, 1020), (1985, 1060), (380, 1060), (380, 1110)])
    text(1040, 1050, "per-service measurements + effective execution observations", size=19, fill=BLUE)

    # Stages 5-7: measurement, reproducibility records and durable catalogue.
    panel(80, 1110, 600, 400, "05", "Service metadata / QoS", "Extract typed observations and measurement context")
    group(105, 1235, "FUNCTIONAL DESCRIPTION", ["Task, modality, model, data and interface schemas", "Family membership and realised training distribution"], size=21)
    group(105, 1330, "QUALITY + QOS + RESOURCES", ["Named task metric, loss and normalised quality score", "Runtime / inference timing, memory and model size", "Reliability indicators and modelled resource cost"], size=21)
    group(105, 1454, "EVIDENCE SEMANTICS", ["Units, direction, protocol, counts, intervals and status"], size=21)

    panel(780, 1110, 600, 400, "06", "Provenance / reconstruction", "Persist authoritative recipes after execution", accent=PURPLE)
    group(805, 1235, "PER-CLIENT RECONSTRUCTION RECIPE", ["Family / initial-state hashes; ordered accepted row IDs", "Effective settings, seeds, updates and stopping reason", "Prepared-content hashes, metrics and runtime versions"], accent=PURPLE, size=20)
    group(805, 1352, "EXECUTION EVIDENCE", ["Requested-to-effective settings and runtime changes", "Hashed append-only execution records + artifact links"], accent=PURPLE, size=20)
    rect(805, 1417, 550, 70, fill="#F0EBF7", stroke=None, radius=9)
    lines(820, 1444, ["Default: retain shared w0 + recipes; release client weights", "Optional: retain checkpoints / validation references"], size=20, leading=27, fill=PURPLE, max_width=520)

    panel(1480, 1110, 600, 400, "07", "Benchmark database", "SQLite catalogue linked to durable artifacts", accent=PURPLE)
    group(1505, 1235, "QUERYABLE RECORDS", ["services  |  service_metrics  |  service_artifacts", "Generation, split and execution provenance", "Lineage / retention records; explicit failure evidence"], accent=PURPLE, size=21)
    group(1505, 1352, "CONTENT-ADDRESSED ARTIFACT STORE", ["Family documents, common initial checkpoints", "Recipes, audit reservations and evaluation evidence"], accent=PURPLE, size=21)
    rect(1505, 1417, 550, 70, fill="#F0EBF7", stroke=None, radius=9)
    lines(1520, 1444, ["Join by service / family IDs; verify references and hashes", "Resume skips verified completed services"], size=20, leading=27, fill=PURPLE, max_width=520)
    path([(680, 1320), (780, 1320)])
    path([(1380, 1320), (1480, 1320)])
    path([(1775, 1510), (1775, 1580)])
    text(1810, 1552, "catalogue + verified artifact links", size=17, fill=PURPLE, max_width=270)

    # Stage 8: intended reuse, distinguishing catalogue search from audit execution.
    panel(80, 1580, 2000, 277, "08", "Downstream benchmark tasks", "Supported workflows and proposed reuse protocols; experiments require their own evaluation", accent=ORANGE)
    for x in (105, 767, 1429):
        rect(x, 1681, 626, 151, fill="#FCF5EC", stroke=None, radius=10)
    group(124, 1711, "CATALOGUE ANALYSIS", ["Service filtering / ranking and QoS prediction", "Evidence coverage, failure and diversity analysis", "Use held-out families and prevent target leakage"], width=588, accent=ORANGE, size=21)
    group(786, 1711, "RECONSTRUCTION BENCHMARKS", ["Verify dependencies; replay selected client recipes", "Measure fidelity against a retained reference / twin", "Report storage, replay cost and functional agreement"], width=588, accent=ORANGE, size=21)
    group(1448, 1711, "COMPATIBLE COLLABORATIVE COMPOSITION", ["Freeze requests / candidates; MAB, GA, greedy, random", "Replay members; sample-weighted parameter averaging", "Measure aggregate quality / latency on saved audit rows"], width=588, accent=ORANGE, size=20)
    path([(2120, 1750), (2046, 1750)], stroke=TEAL, dashed=True)

    # Concise legend and execution-order qualification.
    path([(80, 1896), (145, 1896)], width=2.5)
    text(158, 1903, "Primary data flow", size=18, fill=MUTED)
    path([(360, 1896), (425, 1896)], stroke=TEAL, dashed=True, width=2.5)
    text(438, 1903, "Shared dependencies / reserved audit data", size=18, fill=MUTED)
    text(950, 1903, "Recipe bands describe planned conditions; observed quality can be weak.", size=18, fill=MUTED)
    text(80, 1941, "Logical framework: hf-manifest plans contracts and recipes; run-manifest freezes durable families and executes clients. Replay fidelity is evaluated separately.", size=18, fill=MUTED, max_width=2000)

    c.showPage()
    c.save()
    svg.append("</svg>")
    svg_path = OUT / "dataset-generator-new-mode-framework.svg"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    print(json.dumps({"pdf": str(PDF), "svg": str(svg_path), "text_width_checks": len(checks)}, indent=2))


if __name__ == "__main__":
    build()
