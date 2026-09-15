#!/usr/bin/env python3
"""Build an editable PowerPoint deck for the MLaaS / MSC presentation."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# Theme (matches prior Marp deck)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
ACCENT_DARK = RGBColor(0x1E, 0x3A, 0x8A)
ACCENT_LIGHT = RGBColor(0xDB, 0xEA, 0xFE)
MUTED = RGBColor(0x64, 0x74, 0x8B)
TEXT = RGBColor(0x0F, 0x17, 0x2A)
SUBTEXT = RGBColor(0x33, 0x41, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_TEXT = RGBColor(0xCB, 0xD5, 0xE1)
SURFACE = RGBColor(0xF8, 0xFA, 0xFC)
CODE_BG = RGBColor(0x0F, 0x17, 0x2A)
CODE_FG = RGBColor(0xE2, 0xE8, 0xF0)

# Implementation architecture pipeline colours (reference slide)
ARCH_REGISTRY = RGBColor(0x3B, 0x82, 0xF6)
ARCH_MANIFEST = RGBColor(0x22, 0xC5, 0x5E)
ARCH_EXECUTION = RGBColor(0xF9, 0x73, 0x16)
ARCH_METRICS = RGBColor(0xA8, 0x55, 0xF7)
ARCH_DATASET = RGBColor(0x1E, 0x3A, 0x8A)
ARCH_BANNER_LIGHT = RGBColor(0xF1, 0xF5, 0xF9)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN_L = Inches(0.55)
MARGIN_R = Inches(0.55)
CONTENT_W = SLIDE_W - MARGIN_L - MARGIN_R
FOOTER_Y = Inches(7.05)

OUTPUT = Path(__file__).resolve().parent / "MLaaS_presentation.pptx"


def _blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _set_slide_size(prs: Presentation) -> None:
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H


def _add_rect(slide, left, top, width, height, fill_rgb, line_rgb=None):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if line_rgb is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_rgb
    return shape


def _add_textbox(
    slide,
    left,
    top,
    width,
    height,
    text: str,
    *,
    font_size=18,
    bold=False,
    color=TEXT,
    align=PP_ALIGN.LEFT,
    font_name="Segoe UI",
    valign=MSO_ANCHOR.TOP,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.name = font_name
    p.font.color.rgb = color
    return box


def _add_footer(slide, text: str, light: bool = False) -> None:
    _add_textbox(
        slide,
        MARGIN_L,
        FOOTER_Y,
        CONTENT_W,
        Inches(0.35),
        text,
        font_size=9,
        color=LIGHT_TEXT if light else MUTED,
        align=PP_ALIGN.LEFT,
    )


def _add_title_block(
    slide,
    title: str,
    subtitle: str | None = None,
    *,
    y=Inches(0.45),
    accent_bar: bool = True,
) -> float:
    if accent_bar:
        _add_rect(slide, MARGIN_L, y, Inches(0.08), Inches(0.72), ACCENT)
        title_x = MARGIN_L + Inches(0.2)
    else:
        title_x = MARGIN_L
    _add_textbox(
        slide,
        title_x,
        y,
        CONTENT_W - Inches(0.2),
        Inches(0.65),
        title,
        font_size=32,
        bold=True,
        color=ACCENT_DARK,
    )
    next_y = y + Inches(0.78)
    if subtitle:
        _add_textbox(
            slide,
            MARGIN_L,
            next_y,
            CONTENT_W,
            Inches(0.35),
            subtitle,
            font_size=14,
            bold=True,
            color=ACCENT,
        )
        next_y += Inches(0.42)
    return next_y


def _add_bullets(slide, left, top, width, height, items: list[str], font_size=16) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.size = Pt(font_size)
        p.font.name = "Segoe UI"
        p.font.color.rgb = TEXT
        p.space_after = Pt(6)


def _add_table(slide, left, top, width, rows: list[list[str]], col_widths: list[float] | None = None):
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    height = Inches(0.38 * n_rows + 0.25)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table

    if col_widths:
        for ci, frac in enumerate(col_widths):
            table.columns[ci].width = int(width * frac)

    for ri, row in enumerate(rows):
        for ci, cell_text in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = cell_text
            for p in cell.text_frame.paragraphs:
                p.font.name = "Segoe UI"
                p.font.size = Pt(11 if ri > 0 or ci > 0 else 12)
                if ri == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                else:
                    p.font.color.rgb = TEXT
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT
            elif ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = SURFACE
    return table_shape


def _add_flow_bar(slide, steps: list[str], top: float) -> float:
    bar_h = Inches(0.55)
    _add_rect(slide, MARGIN_L, top, CONTENT_W, bar_h, CODE_BG)
    parts = []
    for i, step in enumerate(steps):
        if i:
            parts.append("  →  ")
        parts.append(step)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.15),
        top + Inches(0.1),
        CONTENT_W - Inches(0.3),
        bar_h,
        "".join(parts),
        font_size=13,
        color=CODE_FG,
        align=PP_ALIGN.CENTER,
        font_name="Consolas",
    )
    return top + bar_h + Inches(0.2)


def _add_compact_card(
    slide,
    left,
    top,
    width,
    height,
    title: str,
    bullets: list[str],
    *,
    highlight: bool = False,
    font_size: int = 10,
) -> None:
    border = ACCENT if highlight else RGBColor(0xE2, 0xE8, 0xF0)
    fill = ACCENT_LIGHT if highlight else SURFACE
    _add_rect(slide, left, top, width, height, fill, border)
    if highlight:
        _add_rect(slide, left, top, Inches(0.07), height, ACCENT)
    title_x = left + (Inches(0.12) if highlight else Inches(0.1))
    _add_textbox(
        slide,
        title_x,
        top + Inches(0.08),
        width - Inches(0.2),
        Inches(0.32),
        title,
        font_size=11 if highlight else 10,
        bold=True,
        color=ACCENT_DARK,
    )
    _add_bullets(
        slide,
        title_x,
        top + Inches(0.4),
        width - Inches(0.2),
        height - Inches(0.48),
        bullets,
        font_size=font_size,
    )


def _add_highlight_banner(slide, top, text: str) -> float:
    h = Inches(0.62)
    _add_rect(slide, MARGIN_L, top, CONTENT_W, h, ACCENT_DARK)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.2),
        top + Inches(0.12),
        CONTENT_W - Inches(0.4),
        h - Inches(0.2),
        text,
        font_size=12,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    return top + h + Inches(0.1)


def _add_heterogeneity_pipeline(slide, left, top, width, height) -> None:
    steps = [
        "Base models",
        "Different datasets",
        "Training conditions",
        "Runtime behaviours",
        "ML service records",
    ]
    n = len(steps)
    gap = Inches(0.08)
    box_h = (height - gap * (n - 1)) / n
    for i, label in enumerate(steps):
        y = top + i * (box_h + gap)
        fill = ACCENT if i == n - 1 else SURFACE
        text_color = WHITE if i == n - 1 else ACCENT_DARK
        border = ACCENT if i in {0, n - 1} else RGBColor(0xE2, 0xE8, 0xF0)
        _add_rect(slide, left, y, width, box_h, fill, border)
        _add_textbox(
            slide,
            left + Inches(0.08),
            y + Inches(0.06),
            width - Inches(0.16),
            box_h - Inches(0.1),
            label,
            font_size=10,
            bold=(i == 0 or i == n - 1),
            color=text_color,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        if i < n - 1:
            arrow_y = y + box_h
            _add_textbox(
                slide,
                left,
                arrow_y,
                width,
                gap,
                "▼",
                font_size=9,
                color=ACCENT,
                align=PP_ALIGN.CENTER,
            )


def _add_two_cards(
    slide,
    top,
    cards: list[tuple[str, list[str]]],
    card_h=Inches(2.35),
) -> float:
    gap = Inches(0.25)
    card_w = (CONTENT_W - gap) / 2
    for idx, (heading, bullets) in enumerate(cards):
        left = MARGIN_L + idx * (card_w + gap)
        _add_rect(slide, left, top, card_w, card_h, SURFACE, RGBColor(0xE2, 0xE8, 0xF0))
        _add_textbox(
            slide,
            left + Inches(0.15),
            top + Inches(0.12),
            card_w - Inches(0.3),
            Inches(0.35),
            heading,
            font_size=15,
            bold=True,
            color=ACCENT_DARK,
        )
        _add_bullets(
            slide,
            left + Inches(0.15),
            top + Inches(0.5),
            card_w - Inches(0.3),
            card_h - Inches(0.6),
            bullets,
            font_size=13,
        )
    return top + card_h + Inches(0.15)


def _add_stat_row(slide, top, stats: list[tuple[str, str]]) -> float:
    n = len(stats)
    gap = Inches(0.2)
    box_w = (CONTENT_W - gap * (n - 1)) / n
    for i, (value, label) in enumerate(stats):
        left = MARGIN_L + i * (box_w + gap)
        _add_textbox(
            slide,
            left,
            top,
            box_w,
            Inches(0.55),
            value,
            font_size=40,
            bold=True,
            color=ACCENT,
            align=PP_ALIGN.CENTER,
        )
        _add_textbox(
            slide,
            left,
            top + Inches(0.52),
            box_w,
            Inches(0.35),
            label,
            font_size=12,
            color=MUTED,
            align=PP_ALIGN.CENTER,
        )
    return top + Inches(0.95)


def slide_title(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, ACCENT_DARK)

    _add_textbox(
        slide,
        Inches(0.9),
        Inches(1.85),
        Inches(11.5),
        Inches(1.2),
        "Benchmark Dataset Generator for\nMLaaS Ecosystems",
        font_size=36,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide,
        Inches(0.9),
        Inches(3.15),
        Inches(11.5),
        Inches(0.55),
        "MSC: A Benchmark Dataset for Machine Learning as a Service (MLaaS) Composition",
        font_size=16,
        color=LIGHT_TEXT,
        align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide,
        Inches(0.9),
        Inches(4.0),
        Inches(11.5),
        Inches(0.4),
        "Joshua Boland · MATH3004 Industrial Project",
        font_size=14,
        color=LIGHT_TEXT,
        align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide,
        Inches(0.9),
        Inches(4.45),
        Inches(11.5),
        Inches(0.4),
        "BSc Data Science · Supervisors: A/Prof Sajib Mistry & Deepak Kanneganti",
        font_size=13,
        color=LIGHT_TEXT,
        align=PP_ALIGN.CENTER,
    )
    _add_footer(slide, "MSC Benchmark · MATH3004 · ICSOC 2026", light=True)


def slide_motivation(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "Motivation", "IoT, MLaaS, and service composition")
    _add_bullets(
        slide,
        MARGIN_L,
        y,
        CONTENT_W,
        Inches(1.55),
        [
            "MLaaS lets organisations use cloud ML without owning full infrastructure (SageMaker, Azure ML, Hugging Face).",
            "Complex applications often need multiple ML services—preprocessing, classification, summarisation—in one pipeline.",
            "MLaaS composition selects and combines services under accuracy, latency, cost, and reliability constraints.",
        ],
    )
    quote_y = y + Inches(1.65)
    _add_rect(slide, MARGIN_L, quote_y, Inches(0.06), Inches(0.95), ACCENT)
    _add_rect(slide, MARGIN_L + Inches(0.06), quote_y, CONTENT_W - Inches(0.06), Inches(0.95), SURFACE)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.2),
        quote_y + Inches(0.12),
        CONTENT_W - Inches(0.35),
        Inches(0.8),
        "Providers expose limited model and QoS detail. Research lacks public benchmarks with requirements and composition solutions—unlike QWS and WS-DREAM for web services.",
        font_size=14,
        color=SUBTEXT,
    )
    _add_textbox(
        slide,
        MARGIN_L,
        quote_y + Inches(1.05),
        CONTENT_W,
        Inches(0.3),
        "Proposal §1 · ICSOC 2026 §1 (Kanneganti et al.; Al-Masri & Mahmoud 2008; Zheng et al. 2013)",
        font_size=10,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_research_gap(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "Research gap", "Why web-service datasets are not enough")
    _add_table(
        slide,
        MARGIN_L,
        y,
        CONTENT_W,
        [
            ["Dataset", "MLaaS", "Functional", "Rich QoS", "Internal", "XAI", "Composition"],
            ["QWS", "✗", "✗", "◐", "✗", "✗", "✗"],
            ["WS-DREAM", "✗", "✗", "◐", "✗", "✗", "✗"],
            ["HSC", "◐", "✓", "◐", "◐", "✗", "sequential only"],
            ["MSC (ours)", "✓", "✓", "✓", "✓", "✓", "✓"],
        ],
        col_widths=[0.18, 0.1, 0.12, 0.12, 0.12, 0.1, 0.26],
    )
    _add_textbox(
        slide,
        MARGIN_L,
        y + Inches(2.35),
        CONTENT_W,
        Inches(0.3),
        "Adapted from ICSOC 2026 Table 1 · HSC: Wang et al., ICSOC 2024",
        font_size=10,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_objectives(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "Project objectives", "MATH3004 proposal — five aims")
    _add_table(
        slide,
        MARGIN_L,
        y,
        CONTENT_W,
        [
            ["#", "Objective"],
            ["1", "Configurable framework — diverse ML task families, models, and training conditions"],
            ["2", "Training + inference — realistic measurements from commonly used models"],
            ["3", "Functional + QoS metrics — individual services and compositions"],
            ["4", "Evaluate selection/composition algorithms against prior research"],
            ["5", "Reproducibility — documentation and tools to extend datasets"],
        ],
        col_widths=[0.08, 0.92],
    )
    _add_textbox(
        slide,
        MARGIN_L,
        y + Inches(2.55),
        CONTENT_W,
        Inches(0.55),
        "This repository implements the service-generation core; composition requirements and solutions are built on the published MSC workflow.",
        font_size=12,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_msc_glance(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "MSC dataset at a glance")
    y = _add_stat_row(
        slide,
        y,
        [
            ("25,900", "MLaaS services"),
            ("20,000", "composition requirements"),
            ("12", "task families"),
            ("5", "provider sources"),
        ],
    )
    _add_bullets(
        slide,
        MARGIN_L,
        y + Inches(0.15),
        CONTENT_W,
        Inches(1.2),
        [
            "Largest known MLaaS composition dataset — inference and fine-tuned services",
            "10,000 sequential + 10,000 collaborative composition scenarios",
            "Benchmark solutions via MAB-based composability algorithm",
        ],
        font_size=15,
    )
    _add_textbox(
        slide,
        MARGIN_L,
        Inches(6.55),
        CONTENT_W,
        Inches(0.3),
        "ICSOC 2026 abstract & §3 · Release: anonymous.4open.science/r/MLaaS_Dataset_Gen-F3C4",
        font_size=10,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_composition_types(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "MLaaS composition types")
    _add_two_cards(
        slide,
        y,
        [
            (
                "Sequential composition",
                [
                    "Workflow: output of service i → input of service i+1",
                    "Example: medical image → text caption → clinical summary",
                    "Models workflow order and stage-wise QoS aggregation",
                ],
            ),
            (
                "Collaborative composition",
                [
                    "Multiple services merged into one unified capability",
                    "Example: HAR — walk/run service + sitting service",
                    "Models compatibility across functional and QoS dimensions",
                ],
            ),
        ],
    )
    _add_textbox(
        slide,
        MARGIN_L,
        Inches(6.55),
        CONTENT_W,
        Inches(0.3),
        "ICSOC 2026 §1 · Proposal Fig. 1 (MLaaS composition illustration)",
        font_size=10,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_workflow(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "MSC collection workflow")
    steps = [
        "Collect services",
        "Extract attributes",
        "Generate requirements",
        "MAB / GA solutions",
        "MSC dataset",
    ]
    y = _add_flow_bar(slide, steps, y)
    _add_table(
        slide,
        MARGIN_L,
        y,
        CONTENT_W,
        [
            ["Phase", "Output"],
            ["3.1 Services", "Normalized functional + QoS records per provider instance"],
            ["3.2 Requirements", "Application-driven requests (healthcare vs finance profiles)"],
            ["3.3 Composition", "Feasible combinations optimising sequential / collaborative objectives"],
        ],
        col_widths=[0.28, 0.72],
    )
    _add_textbox(
        slide,
        MARGIN_L,
        Inches(6.55),
        CONTENT_W,
        Inches(0.3),
        "ICSOC 2026 Fig. 2 & §3 — Edit workflow diagram in PowerPoint as needed",
        font_size=10,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_attributes(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "What each MLaaS service records", "Key attributes (ICSOC Table 3)")
    _add_table(
        slide,
        MARGIN_L,
        y,
        CONTENT_W,
        [
            ["Symbol", "Attribute", "Role in composition"],
            ["Δ, Dₙ", "Data distribution & size", "Functional compatibility"],
            ["w", "Model parameters", "Alignment across services"],
            ["A", "Accuracy", "Quality objective"],
            ["L", "Latency", "QoS constraint"],
            ["R", "Reliability", "Consistency under load"],
            ["T, C", "Compute time & cost", "Resource efficiency"],
            ["X", "Explainability", "Perturbation sensitivity score"],
        ],
        col_widths=[0.14, 0.36, 0.5],
    )
    _add_textbox(
        slide,
        MARGIN_L,
        y + Inches(2.75),
        CONTENT_W,
        Inches(0.4),
        "Also captured: tail latency, cost efficiency, popularity, modality, training regime, split provenance.",
        font_size=11,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_explainability(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "Explainability as a QoS metric")

    # Row 1: traditional QoS vs ML interpretability
    row1_h = Inches(1.05)
    half_w = (CONTENT_W - Inches(0.2)) / 2
    _add_rect(slide, MARGIN_L, y, half_w, row1_h, SURFACE, RGBColor(0xE2, 0xE8, 0xF0))
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.12),
        y + Inches(0.08),
        half_w - Inches(0.24),
        Inches(0.28),
        "Traditional service QoS",
        font_size=13,
        bold=True,
        color=ACCENT_DARK,
    )
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.12),
        y + Inches(0.38),
        half_w - Inches(0.24),
        Inches(0.62),
        "Latency · reliability · cost · throughput — well defined for web and cloud services (QWS, WS-DREAM).",
        font_size=12,
        color=TEXT,
    )
    right_l = MARGIN_L + half_w + Inches(0.2)
    _add_rect(slide, right_l, y, half_w, row1_h, SURFACE, RGBColor(0xE2, 0xE8, 0xF0))
    _add_textbox(
        slide,
        right_l + Inches(0.12),
        y + Inches(0.08),
        half_w - Inches(0.24),
        Inches(0.28),
        "ML services add interpretability",
        font_size=13,
        bold=True,
        color=ACCENT_DARK,
    )
    _add_textbox(
        slide,
        right_l + Inches(0.12),
        y + Inches(0.38),
        half_w - Inches(0.24),
        Inches(0.62),
        "Users must trust predictions — especially healthcare (clinical decisions) and finance (risk, compliance).",
        font_size=12,
        color=TEXT,
    )
    y += row1_h + Inches(0.15)

    # Row 2: challenge callout
    challenge_h = Inches(0.72)
    _add_rect(slide, MARGIN_L, y, Inches(0.06), challenge_h, ACCENT)
    _add_rect(slide, MARGIN_L + Inches(0.06), y, CONTENT_W - Inches(0.06), challenge_h, ACCENT_LIGHT)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.18),
        y + Inches(0.1),
        CONTENT_W - Inches(0.35),
        challenge_h - Inches(0.15),
        "Challenge: define explainability consistently across 12 task families — text, vision, and tabular each need different XAI tooling.",
        font_size=13,
        bold=True,
        color=ACCENT_DARK,
    )
    y += challenge_h + Inches(0.15)

    # Row 3: rejected approach vs engineering decision
    row3_h = Inches(1.55)
    _add_two_cards(
        slide,
        y,
        [
            (
                "Why not a unified SHAP / Grad-CAM / LIME pipeline?",
                [
                    "Methods are task-specific (tokens vs pixels vs feature columns)",
                    "Heavy models × thousands of services → prohibitive cost",
                    "Would block batch dataset generation on commodity hardware",
                ],
            ),
            (
                "Engineering decision: perturbation-based proxy",
                [
                    "Perturb inputs on benchmark eval samples",
                    "Measure output stability (confidence, prediction, quality)",
                    "Stable behaviour → higher explainability_score (MSC attribute X)",
                ],
            ),
        ],
        card_h=row3_h,
    )
    y += row3_h + Inches(0.12)

    # Row 4: implementation
    impl_h = Inches(1.05)
    _add_rect(slide, MARGIN_L, y, CONTENT_W, impl_h, CODE_BG)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.15),
        y + Inches(0.08),
        CONTENT_W - Inches(0.3),
        Inches(0.28),
        "Implemented in this framework (services/perturbation.py)",
        font_size=12,
        bold=True,
        color=CODE_FG,
    )
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.15),
        y + Inches(0.36),
        CONTENT_W - Inches(0.3),
        Inches(0.62),
        "Post-eval perturbation stage · task-family-aware input units · trust trials (benign noise) + targeted edits · "
        "sample/time budgets · stored as explainability_score in service_metrics — best-effort, never fails the service run",
        font_size=11,
        color=LIGHT_TEXT,
    )
    _add_textbox(
        slide,
        MARGIN_L,
        Inches(6.52),
        CONTENT_W,
        Inches(0.35),
        "Honours-level framing: recognised the XAI gap, chose a scalable cross-task proxy — not a full academic XAI taxonomy.",
        font_size=10,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland · Proposal §4.5 · ICSOC §3.1 (X)")


def slide_methods(prs: Presentation) -> None:
    """Simulating realistic ML providers — heterogeneous service generation."""
    slide = _blank_slide(prs)
    y = _add_title_block(
        slide,
        "Simulating realistic ML providers",
        "Generating heterogeneous ML service behaviour from real model execution",
    )

    content_top = y + Inches(0.05)
    content_h = Inches(4.35)
    pipe_w = Inches(2.15)
    cards_left = MARGIN_L + pipe_w + Inches(0.22)
    cards_w = CONTENT_W - pipe_w - Inches(0.22)

    _add_textbox(
        slide,
        MARGIN_L,
        content_top - Inches(0.02),
        pipe_w,
        Inches(0.28),
        "Heterogeneity pipeline",
        font_size=9,
        bold=True,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    _add_heterogeneity_pipeline(slide, MARGIN_L, content_top + Inches(0.22), pipe_w, content_h - Inches(0.22))

    card_gap = Inches(0.14)
    row_h = (content_h - card_gap) / 2
    col3_w = (cards_w - 2 * card_gap) / 3
    col2_w = (cards_w - card_gap) / 2
    row1_top = content_top
    row2_top = content_top + row_h + card_gap

    cards = [
        (
            "1 · Task & model diversity",
            ["NLP, vision, tabular task families", "Multiple architectures & providers", "Benchmark datasets per task"],
            False,
        ),
        (
            "2 · Service variation",
            [
                "Hyperparameters: LR, batch, epochs, optimiser",
                "IID vs non-IID splits & training skew",
                "Fine-tuned vs inference-only regimes",
            ],
            True,
        ),
        (
            "3 · Runtime execution",
            ["Execute real train / inference workflows", "Measure actual runtime — not synthetic metadata", "CPU / GPU resource tracking"],
            False,
        ),
        (
            "4 · Behavioural measurement",
            [
                "Prediction quality under benchmark eval",
                "Latency, reliability, resource usage",
                "Explainability via perturbation sensitivity",
            ],
            False,
        ),
        (
            "5 · Dataset generation",
            ["Standardised comparable service records", "Reproducible manifest-driven experiments", "Catalogue for composition research"],
            False,
        ),
    ]

    for idx, (title, bullets, highlight) in enumerate(cards[:3]):
        left = cards_left + idx * (col3_w + card_gap)
        _add_compact_card(slide, left, row1_top, col3_w, row_h, title, bullets, highlight=highlight)

    for idx, (title, bullets, highlight) in enumerate(cards[3:]):
        left = cards_left + idx * (col2_w + card_gap)
        _add_compact_card(slide, left, row2_top, col2_w, row_h, title, bullets, highlight=highlight)

    banner_y = content_top + content_h + Inches(0.12)
    _add_highlight_banner(
        slide,
        banner_y,
        "The goal was not to find the best ML model — the goal was to simulate many realistic ML providers with diverse behaviours.",
    )
    _add_footer(slide, "Joshua Boland · Proposal §4 · service ecosystem simulation")


def _add_arch_pipeline_step(
    slide,
    left,
    top,
    width,
    height,
    number: int,
    title: str,
    fill: RGBColor,
) -> None:
    _add_rect(slide, left, top, width, height, fill)
    _add_textbox(
        slide,
        left,
        top + Inches(0.12),
        width,
        Inches(0.38),
        str(number),
        font_size=22,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide,
        left + Inches(0.06),
        top + Inches(0.5),
        width - Inches(0.12),
        height - Inches(0.55),
        title,
        font_size=11,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.TOP,
    )


def _add_arch_detail_column(
    slide,
    left,
    top,
    width,
    height,
    number: int,
    heading: str,
    bullets: list[str],
    accent: RGBColor,
) -> None:
    _add_rect(slide, left, top, width, Inches(0.06), accent)
    _add_rect(slide, left, top + Inches(0.06), width, height - Inches(0.06), SURFACE, RGBColor(0xE2, 0xE8, 0xF0))
    _add_textbox(
        slide,
        left + Inches(0.08),
        top + Inches(0.14),
        width - Inches(0.16),
        Inches(0.32),
        f"{number}. {heading}",
        font_size=10,
        bold=True,
        color=accent,
    )
    _add_bullets(
        slide,
        left + Inches(0.08),
        top + Inches(0.46),
        width - Inches(0.16),
        height - Inches(0.52),
        bullets,
        font_size=9,
    )


def slide_implementation_architecture(prs: Presentation) -> None:
    """Five-stage implementation architecture (replaces separate implementation + pipeline slides)."""
    slide = _blank_slide(prs)
    y = _add_title_block(
        slide,
        "Implementation architecture",
        "From configuration to a benchmark-ready dataset",
    )

    pipe_top = y + Inches(0.05)
    pipe_h = Inches(0.88)
    arrow_w = Inches(0.22)
    n_steps = 5
    step_w = (CONTENT_W - arrow_w * (n_steps - 1)) / n_steps
    steps = [
        ("Registry", ARCH_REGISTRY),
        ("Manifest\ngeneration", ARCH_MANIFEST),
        ("Service\nexecution", ARCH_EXECUTION),
        ("Metrics\ncollection", ARCH_METRICS),
        ("Benchmark\ndataset", ARCH_DATASET),
    ]
    x = MARGIN_L
    for i, (label, color) in enumerate(steps):
        _add_arch_pipeline_step(slide, x, pipe_top, step_w, pipe_h, i + 1, label.replace("\n", " "), color)
        x += step_w
        if i < n_steps - 1:
            _add_textbox(
                slide,
                x,
                pipe_top + Inches(0.28),
                arrow_w,
                Inches(0.35),
                "→",
                font_size=18,
                bold=True,
                color=MUTED,
                align=PP_ALIGN.CENTER,
            )
            x += arrow_w

    detail_top = pipe_top + pipe_h + Inches(0.18)
    detail_h = Inches(2.35)
    col_gap = Inches(0.12)
    col_w = (CONTENT_W - col_gap * (n_steps - 1)) / n_steps
    details = [
        (
            "Registry",
            ARCH_REGISTRY,
            [
                "Curated tasks, models, and datasets",
                "Hugging Face and generic providers",
                "NLP, vision, tabular, detection, segmentation, …",
            ],
        ),
        (
            "Manifest generation",
            ARCH_MANIFEST,
            [
                "hf-manifest builds many service rows",
                "Vary hyperparameters, datasets, IID / non-IID splits",
                "Training regimes & knob variants at scale",
            ],
        ),
        (
            "Service execution",
            ARCH_EXECUTION,
            [
                "run-manifest: validate → train / inference",
                "Loads splits, builds model, runs benchmark eval",
                "CPU / GPU execution · grouped HF · multi-worker",
            ],
        ),
        (
            "Metrics collection",
            ARCH_METRICS,
            [
                "Behaviour: accuracy, F1, task metrics, metric_score",
                "QoS: latency, reliability, resource, cost",
                "Explainability score (perturbation stage)",
            ],
        ),
        (
            "Benchmark dataset",
            ARCH_DATASET,
            [
                "SQLite services.db — normalised schema",
                "One manifest row → one service record",
                "Ready for composition & algorithm evaluation",
            ],
        ),
    ]
    for i, (heading, accent, bullets) in enumerate(details):
        left = MARGIN_L + i * (col_w + col_gap)
        _add_arch_detail_column(slide, left, detail_top, col_w, detail_h, i + 1, heading, bullets, accent)

    dark_banner_y = detail_top + detail_h + Inches(0.14)
    dark_h = Inches(0.72)
    _add_rect(slide, MARGIN_L, dark_banner_y, CONTENT_W, dark_h, ACCENT_DARK)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.25),
        dark_banner_y + Inches(0.1),
        CONTENT_W - Inches(0.5),
        Inches(0.32),
        "One manifest row = one simulated ML provider",
        font_size=14,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.25),
        dark_banner_y + Inches(0.4),
        CONTENT_W - Inches(0.5),
        Inches(0.28),
        "Each row defines a unique provider configuration; execution produces one record in the benchmark catalogue.",
        font_size=10,
        color=LIGHT_TEXT,
        align=PP_ALIGN.CENTER,
    )

    light_banner_y = dark_banner_y + dark_h + Inches(0.1)
    light_h = Inches(0.48)
    _add_rect(slide, MARGIN_L, light_banner_y, CONTENT_W, light_h, ARCH_BANNER_LIGHT, RGBColor(0xE2, 0xE8, 0xF0))
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.2),
        light_banner_y + Inches(0.1),
        CONTENT_W - Inches(0.4),
        light_h - Inches(0.15),
        "This framework generates heterogeneous, realistic ML service providers and captures their behaviour for reproducible benchmarking.",
        font_size=10,
        color=SUBTEXT,
        align=PP_ALIGN.CENTER,
    )
    _add_footer(slide, "Joshua Boland · registry → hf-manifest → run-manifest → services.db")


def slide_tasks(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(
        slide,
        "Why task diversity matters",
        "Different ML tasks produce different runtime, QoS, and explainability behaviour",
    )

    intro_h = Inches(0.58)
    _add_rect(slide, MARGIN_L, y, Inches(0.06), intro_h, ACCENT)
    _add_rect(slide, MARGIN_L + Inches(0.06), y, CONTENT_W - Inches(0.06), intro_h, SURFACE)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.18),
        y + Inches(0.1),
        CONTENT_W - Inches(0.35),
        intro_h - Inches(0.15),
        "Web-service benchmarks cannot assume one QoS profile. ML tasks differ in compute, latency, memory, and trust signals — richer ML-specific measurement is required.",
        font_size=12,
        color=SUBTEXT,
    )
    y += intro_h + Inches(0.14)

    content_h = Inches(3.55)
    card_gap = Inches(0.14)
    row_h = (content_h - card_gap) / 2
    col3_w = (CONTENT_W - 2 * card_gap) / 3
    col2_w = (CONTENT_W - card_gap) / 2
    row2_offset = (CONTENT_W - (2 * col2_w + card_gap)) / 2

    domain_cards = [
        (
            "NLP",
            [
                "Variable sequence lengths → latency spread",
                "Token-level vs sentence-level workloads",
                "Different explainability (perturbation on text units)",
            ],
            RGBColor(0x0E, 0x74, 0x90),
        ),
        (
            "Vision",
            [
                "High compute & GPU cost per sample",
                "Batch size strongly affects throughput",
                "Spatial explainability differs from text",
            ],
            RGBColor(0x7C, 0x3A, 0xED),
        ),
        (
            "Segmentation",
            [
                "Large memory & VRAM footprint",
                "Longer runtimes than classification",
                "Pixel-level outputs → distinct QoS trade-offs",
            ],
            RGBColor(0xDB, 0x27, 0x77),
        ),
        (
            "Classification",
            [
                "Often faster inference per sample",
                "Accuracy-sensitive provider ranking",
                "Reliability varies with class imbalance / skew",
            ],
            RGBColor(0x05, 0x96, 0x69),
        ),
        (
            "Regression / tabular",
            [
                "Lower compute than deep vision/NLP",
                "Feature-space behaviour, not sequences",
                "Different error metrics (RMSE, MAE) & cost profiles",
            ],
            RGBColor(0xD9, 0x77, 0x06),
        ),
    ]

    for idx, (title, bullets, accent) in enumerate(domain_cards[:3]):
        left = MARGIN_L + idx * (col3_w + card_gap)
        _add_rect(slide, left, y, col3_w, Inches(0.05), accent)
        _add_compact_card(slide, left, y + Inches(0.05), col3_w, row_h - Inches(0.05), title, bullets, font_size=10)

    row2_y = y + row_h + card_gap
    for idx, (title, bullets, accent) in enumerate(domain_cards[3:]):
        left = MARGIN_L + row2_offset + idx * (col2_w + card_gap)
        _add_rect(slide, left, row2_y, col2_w, Inches(0.05), accent)
        _add_compact_card(slide, left, row2_y + Inches(0.05), col2_w, row_h - Inches(0.05), title, bullets, font_size=10)

    banner_y = row2_y + row_h + Inches(0.14)
    _add_highlight_banner(
        slide,
        banner_y,
        "The goal was not simply to include many tasks — the goal was to generate realistic variation in ML service behaviour.",
    )
    _add_textbox(
        slide,
        MARGIN_L,
        banner_y + Inches(0.72),
        CONTENT_W,
        Inches(0.28),
        "Generator spans 12 MSC task families (NLP, vision, tabular, detection, generation, …) via hf-manifest — scale supports diversity, not a model inventory.",
        font_size=9,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    _add_footer(slide, "Joshua Boland · ICSOC §3.1 · behavioural diversity")


def _add_mini_flow_column(slide, left, top, width, height, steps: list[str]) -> None:
    n = len(steps)
    gap = Inches(0.06)
    box_h = (height - gap * (n - 1) - Inches(0.2) * (n - 1)) / n
    y = top
    for i, label in enumerate(steps):
        _add_rect(slide, left, y, width, box_h, ACCENT_LIGHT, ACCENT)
        _add_textbox(
            slide,
            left + Inches(0.06),
            y + Inches(0.04),
            width - Inches(0.12),
            box_h - Inches(0.06),
            label,
            font_size=9,
            color=ACCENT_DARK,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        y += box_h
        if i < n - 1:
            _add_textbox(
                slide,
                left,
                y,
                width,
                Inches(0.18),
                "↓",
                font_size=10,
                color=ACCENT,
                align=PP_ALIGN.CENTER,
            )
            y += Inches(0.18)


def slide_downstream_research(prs: Presentation) -> None:
    """Composition + benchmarking as application layers (replaces requirements + benchmark slides)."""
    slide = _blank_slide(prs)
    y = _add_title_block(
        slide,
        "Downstream research on the service catalogue",
        "Core contribution: dataset generation · Composition & benchmarking as application layers",
    )

    col_gap = Inches(0.22)
    left_w = Inches(4.35)
    right_w = CONTENT_W - left_w - col_gap
    left_l = MARGIN_L
    right_l = left_l + left_w + col_gap
    body_top = y + Inches(0.05)
    body_h = Inches(3.55)

    # --- Left: composition extension ---
    _add_textbox(
        slide,
        left_l,
        body_top,
        left_w,
        Inches(0.3),
        "Extending the dataset to composition research",
        font_size=13,
        bold=True,
        color=ACCENT_DARK,
    )
    flow_top = body_top + Inches(0.38)
    flow_h = Inches(1.35)
    _add_mini_flow_column(
        slide,
        left_l,
        flow_top,
        Inches(1.55),
        flow_h,
        ["Service records", "Requirements", "Applications"],
    )
    _add_bullets(
        slide,
        left_l + Inches(1.65),
        flow_top,
        left_w - Inches(1.7),
        flow_h,
        [
            "Requirements: accuracy needs, latency limits, cost constraints",
            "Applications: recommendation, orchestration, service selection, benchmark evaluation",
        ],
        font_size=10,
    )
    _add_textbox(
        slide,
        left_l,
        flow_top + flow_h + Inches(0.1),
        left_w,
        Inches(0.55),
        "MSC paper adds requirement–solution pairs on top of the catalogue — not produced by the generator CLI today.",
        font_size=9,
        color=MUTED,
    )

    # --- Right: benchmarking ---
    _add_textbox(
        slide,
        right_l,
        body_top,
        right_w,
        Inches(0.3),
        "Benchmarking ML service behaviour",
        font_size=13,
        bold=True,
        color=ACCENT_DARK,
    )
    table_top = body_top + Inches(0.38)
    _add_table(
        slide,
        right_l,
        table_top,
        right_w,
        [
            ["", "Traditional service datasets", "MLaaS dataset generator"],
            ["QoS depth", "Limited QoS", "Rich behavioural metrics"],
            ["Behaviour", "Static / advertised", "Measured runtime variability"],
            ["ML awareness", "Minimal", "Accuracy + explainability + resources"],
            ["Provider space", "Small, homogeneous", "Realistic heterogeneous providers"],
        ],
        col_widths=[0.22, 0.39, 0.39],
    )
    enables_top = table_top + Inches(1.55)
    _add_textbox(
        slide,
        right_l,
        enables_top,
        right_w,
        Inches(0.28),
        "The dataset enables:",
        font_size=10,
        bold=True,
        color=ACCENT,
    )
    _add_bullets(
        slide,
        right_l,
        enables_top + Inches(0.28),
        right_w,
        Inches(0.75),
        [
            "Fair comparison across provider configurations",
            "Reproducible evaluation from manifest → execution → records",
            "Realistic optimisation problems (not toy QoS tables)",
        ],
        font_size=10,
    )

    banner_y = body_top + body_h + Inches(0.12)
    _add_highlight_banner(
        slide,
        banner_y,
        "Once realistic ML service records exist, they can support higher-level research like composition and orchestration.",
    )
    _add_footer(slide, "Joshua Boland · ICSOC MSC layer is application research on this catalogue")


def slide_contribution(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "Key contribution")

    statement_h = Inches(0.95)
    _add_rect(slide, MARGIN_L, y, CONTENT_W, statement_h, ACCENT_LIGHT, ACCENT)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.2),
        y + Inches(0.18),
        CONTENT_W - Inches(0.4),
        statement_h - Inches(0.3),
        "A reproducible framework for generating heterogeneous ML service ecosystems.",
        font_size=18,
        bold=True,
        color=ACCENT_DARK,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )
    y += statement_h + Inches(0.2)

    _add_textbox(
        slide,
        MARGIN_L,
        y,
        CONTENT_W,
        Inches(0.32),
        "Enables",
        font_size=14,
        bold=True,
        color=ACCENT,
    )
    y += Inches(0.38)

    enables = [
        "Fair benchmarking across diverse provider configurations",
        "Reproducible evaluation (manifest → execution → standardised records)",
        "Runtime QoS analysis from measured train/inference behaviour",
        "ML service orchestration and selection research",
        "Future composition research on top of the service catalogue",
    ]
    col_gap = Inches(0.2)
    col_w = (CONTENT_W - col_gap) / 2
    mid = (len(enables) + 1) // 2
    for col, chunk in enumerate([enables[:mid], enables[mid:]]):
        left = MARGIN_L + col * (col_w + col_gap)
        _add_bullets(slide, left, y, col_w, Inches(2.35), chunk, font_size=13)
    _add_footer(slide, "Joshua Boland · MATH3004 + ICSOC MSC")


def slide_limitations(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(
        slide,
        "Limitations and open challenges",
        "Critical reflection on trade-offs and future work",
    )
    card_gap = Inches(0.18)
    card_w = (CONTENT_W - card_gap) / 2
    card_h = Inches(2.05)
    cards = [
        (
            "Explainability",
            "No universal XAI metric across all ML task families — perturbation proxy is a deliberate engineering trade-off.",
        ),
        (
            "Runtime cost",
            "Large-scale evaluation is computationally expensive; sample caps and time budgets are required in practice.",
        ),
        (
            "Real-world realism",
            "Public Hugging Face and open models may not fully reflect commercial MLaaS provider behaviour.",
        ),
        (
            "Ecosystem evolution",
            "Benchmark catalogues must evolve as models, tasks, and provider offerings change over time.",
        ),
    ]
    for i, (title, body) in enumerate(cards):
        col = i % 2
        row = i // 2
        left = MARGIN_L + col * (card_w + card_gap)
        top = y + row * (card_h + card_gap)
        _add_rect(slide, left, top, card_w, Inches(0.05), MUTED)
        _add_rect(slide, left, top + Inches(0.05), card_w, card_h - Inches(0.05), SURFACE, RGBColor(0xE2, 0xE8, 0xF0))
        _add_textbox(
            slide,
            left + Inches(0.12),
            top + Inches(0.14),
            card_w - Inches(0.24),
            Inches(0.3),
            title,
            font_size=12,
            bold=True,
            color=ACCENT_DARK,
        )
        _add_textbox(
            slide,
            left + Inches(0.12),
            top + Inches(0.48),
            card_w - Inches(0.24),
            card_h - Inches(0.55),
            body,
            font_size=10,
            color=SUBTEXT,
        )
    _add_footer(slide, "Joshua Boland")


def slide_demo(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    y = _add_title_block(slide, "Demo — reproduce services locally")
    code = """python -m mlaas_data_generator.cli.main hf-manifest \\
  --manifest-profile test \\
  --total-services 8 --output outputs/service_manifest.xlsx

python -m mlaas_data_generator.cli.main run-manifest \\
  --file outputs/service_manifest.xlsx --dry-run

python -m mlaas_data_generator.cli.main run-manifest \\
  --file outputs/service_manifest.xlsx --db outputs/services.db"""
    _add_rect(slide, MARGIN_L, y, CONTENT_W, Inches(2.35), CODE_BG)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.2),
        y + Inches(0.15),
        CONTENT_W - Inches(0.4),
        Inches(2.1),
        code,
        font_size=12,
        color=CODE_FG,
        font_name="Consolas",
    )
    _add_textbox(
        slide,
        MARGIN_L,
        y + Inches(2.45),
        CONTENT_W,
        Inches(0.35),
        "Scale path: test → balanced → benchmark profiles; multi-GPU via --workers",
        font_size=11,
        color=MUTED,
    )
    _add_footer(slide, "Joshua Boland")


def slide_summary(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, ACCENT_DARK)
    _add_textbox(
        slide,
        Inches(0.9),
        Inches(0.5),
        Inches(11.5),
        Inches(0.55),
        "Summary",
        font_size=34,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )

    rows = [
        ("Problem", "No realistic benchmark for heterogeneous ML service behaviour at scale"),
        ("Solution", "A configurable ML service dataset generator"),
        ("Method", "Execute diverse ML providers and capture runtime behaviour"),
        ("Output", "Standardised benchmark-ready service records"),
        (
            "Contribution",
            "Supports reproducible MLaaS benchmarking and future orchestration research",
        ),
    ]
    top = Inches(1.15)
    row_h = Inches(0.72)
    label_w = Inches(1.45)
    for i, (label, text) in enumerate(rows):
        t = top + i * (row_h + Inches(0.08))
        _add_textbox(
            slide,
            MARGIN_L + Inches(0.15),
            t,
            label_w,
            row_h,
            label,
            font_size=12,
            bold=True,
            color=ACCENT_LIGHT,
        )
        _add_textbox(
            slide,
            MARGIN_L + label_w + Inches(0.1),
            t,
            CONTENT_W - label_w - Inches(0.25),
            row_h,
            text,
            font_size=12,
            color=WHITE,
        )

    close_y = top + len(rows) * (row_h + Inches(0.08)) + Inches(0.15)
    _add_rect(slide, MARGIN_L + Inches(0.35), close_y, CONTENT_W - Inches(0.7), Inches(0.7), ACCENT)
    _add_textbox(
        slide,
        MARGIN_L + Inches(0.5),
        close_y + Inches(0.12),
        CONTENT_W - Inches(1.0),
        Inches(0.48),
        "The project shifts benchmarking from static web services toward realistic, behaviour-driven ML service ecosystems.",
        font_size=13,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    _add_textbox(
        slide,
        Inches(0.9),
        Inches(6.55),
        Inches(11.5),
        Inches(0.4),
        "Questions?",
        font_size=26,
        bold=True,
        color=LIGHT_TEXT,
        align=PP_ALIGN.CENTER,
    )
    _add_footer(slide, "Joshua Boland", light=True)


def build(output: Path = OUTPUT) -> Path:
    prs = Presentation()
    _set_slide_size(prs)

    slide_title(prs)
    slide_motivation(prs)
    slide_research_gap(prs)
    slide_objectives(prs)
    slide_msc_glance(prs)
    slide_composition_types(prs)
    slide_workflow(prs)
    slide_attributes(prs)
    slide_explainability(prs)
    slide_methods(prs)
    slide_implementation_architecture(prs)
    slide_tasks(prs)
    slide_downstream_research(prs)
    slide_contribution(prs)
    slide_limitations(prs)
    slide_demo(prs)
    slide_summary(prs)

    try:
        prs.save(output)
    except PermissionError:
        fallback = output.with_name("MLaaS_presentation.generated.pptx")
        prs.save(fallback)
        print(
            f"WARNING: Could not overwrite {output.name} (file may be open in PowerPoint). "
            f"Wrote {fallback} instead.",
            flush=True,
        )
        return fallback
    return output


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")
