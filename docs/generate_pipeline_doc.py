#!/usr/bin/env python3
"""
Generate a formal PDF companion document from the pipeline glossary data.

Usage:
    python3 docs/generate_pipeline_doc.py

Output:
    docs/GSTV_Site_Scoring_Pipeline_Documentation.pdf
"""

import json
import re
from pathlib import Path

from fpdf import FPDF

# ── Constants ────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent
HTML_PATH = DOCS_DIR / "pipeline_glossary.html"
OUTPUT_PATH = DOCS_DIR / "GSTV_Site_Scoring_Pipeline_Documentation.pdf"

# Colors (RGB)
NAVY = (15, 23, 42)
DARK_BLUE = (30, 41, 59)
INDIGO = (99, 102, 241)
SLATE_700 = (51, 65, 85)
SLATE_400 = (148, 163, 184)
WHITE = (255, 255, 255)
NEAR_WHITE = (226, 232, 240)
LIGHT_BG = (241, 245, 249)
EMERALD = (16, 185, 129)
AMBER = (245, 158, 11)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities, replacing non-latin1 chars."""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r"<code>(.*?)</code>", r'"\1"', text)
    text = re.sub(r"<strong>(.*?)</strong>", r"\1", text)
    text = re.sub(r"<em>(.*?)</em>", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&mdash;", " -- ").replace("&ndash;", " - ")
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Unicode replacements for latin-1 safe output
    text = text.replace("\u2014", " -- ").replace("\u2013", " - ")
    text = text.replace("\u00d7", "x")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...").replace("\u2022", "-")
    text = text.replace("\u2192", "->").replace("\u2190", "<-")
    # Final safety: replace any remaining non-latin1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.strip()


def safe_text(text: str) -> str:
    """Ensure text is safe for fpdf latin-1 encoding."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u2013", " - ").replace("\u2014", " -- ")
    text = text.replace("\u00d7", "x")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...").replace("\u2022", "-")
    text = text.replace("\u2192", "->").replace("\u2190", "<-")
    return text.encode("latin-1", errors="replace").decode("latin-1")


def load_stage_data() -> list[dict]:
    """Extract STAGE_DATA JSON from the glossary HTML file."""
    content = HTML_PATH.read_text()
    match = re.search(r"const STAGE_DATA\s*=\s*(\[.*?\]);\s*\n", content, re.DOTALL)
    if not match:
        raise RuntimeError("Could not find STAGE_DATA in pipeline_glossary.html")
    return json.loads(match.group(1))


class PipelineDoc(FPDF):
    """Custom PDF document for the pipeline documentation."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="letter")
        self.set_auto_page_break(auto=True, margin=25)
        self.set_margins(left=25, top=25, right=25)
        # Track current chapter for headers
        self._current_chapter = ""

    def cell(self, *args, **kwargs):
        """Override to sanitize text for latin-1."""
        if args and len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = safe_text(args[2])
        if "text" in kwargs and isinstance(kwargs["text"], str):
            kwargs["text"] = safe_text(kwargs["text"])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        """Override to sanitize text for latin-1."""
        if args and len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = safe_text(args[2])
        if "text" in kwargs and isinstance(kwargs["text"], str):
            kwargs["text"] = safe_text(kwargs["text"])
        return super().multi_cell(*args, **kwargs)

    def header(self):
        if self.page_no() == 1:
            return  # Title page has custom header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*SLATE_400)
        self.cell(0, 8, "GSTV Site Scoring Platform  |  Data Pipeline Documentation", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*SLATE_700)
        self.line(25, 18, self.w - 25, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*SLATE_400)
        self.cell(0, 10, "Confidential  |  GSTV Internal Use", align="C")

    # ── Reusable primitives ─────────────────────────────────────────────

    def section_title(self, text: str, size: int = 16):
        """Large section heading."""
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, text: str, size: int = 12):
        """Smaller subsection heading."""
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*DARK_BLUE)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str, size: int = 10):
        """Standard paragraph text."""
        self.set_font("Helvetica", "", size)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def labeled_block(self, label: str, text: str, label_color=INDIGO):
        """A labeled callout block (Rationale, Technical Note, etc.)."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*label_color)
        self.cell(0, 6, label.upper(), new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*SLATE_700)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def stat_row(self, stats: list[dict]):
        """Render a row of key statistics."""
        if not stats:
            return
        col_w = (self.w - 50) / len(stats)
        y_start = self.get_y()

        for stat in stats:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*INDIGO)
            self.cell(col_w, 7, str(stat.get("value", "")))

        self.set_y(y_start + 7)
        for stat in stats:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*SLATE_400)
            self.cell(col_w, 5, stat.get("label", ""))

        self.ln(10)

    def source_table(self, sources: list[dict]):
        """Render data source details as a table."""
        if not sources:
            return
        self.subsection_title("Data Sources")
        col_widths = [55, 30, 18, 25, 15]  # name, source, format, rows, cols
        headers = ["Dataset", "Source", "Format", "Rows", "Cols"]

        # Header row
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*SLATE_400)
        self.set_fill_color(*LIGHT_BG)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=0, fill=True)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*NAVY)
        for src in sources:
            self.cell(col_widths[0], 6, src.get("name", "")[:30])
            self.cell(col_widths[1], 6, src.get("source", "")[:18])
            self.cell(col_widths[2], 6, src.get("format", ""))
            self.cell(col_widths[3], 6, str(src.get("rows", "")))
            self.cell(col_widths[4], 6, str(src.get("cols", "")))
            self.ln()
        self.ln(3)

        # Detailed notes per source
        for src in sources:
            if src.get("notes"):
                self.set_font("Helvetica", "B", 8)
                self.set_text_color(*DARK_BLUE)
                self.cell(0, 5, src["name"], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 8)
                self.set_text_color(*SLATE_700)
                self.multi_cell(0, 4.5, strip_html(src["notes"]))
                self.ln(2)

    def step_list(self, steps: list[dict]):
        """Render pipeline steps as a numbered list with details."""
        if not steps:
            return
        self.subsection_title("Process Steps")
        for step in steps:
            order = step.get("step_order", 0) + 1
            title = strip_html(step.get("title", ""))
            sub = strip_html(step.get("sub", ""))

            # Step heading
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*NAVY)
            self.cell(0, 7, f"Step {order}: {title}", new_x="LMARGIN", new_y="NEXT")

            # Subtitle
            if sub:
                self.set_font("Helvetica", "I", 9)
                self.set_text_color(*SLATE_700)
                self.multi_cell(0, 5, sub)
                self.ln(1)

            # Description / analogy
            if step.get("analogy"):
                self.set_font("Helvetica", "", 9)
                self.set_text_color(*NAVY)
                self.multi_cell(0, 5, strip_html(step["analogy"]))
                self.ln(1)

            # Why / rationale
            if step.get("why"):
                self.labeled_block("Rationale", strip_html(step["why"]))

            # Technical details
            if step.get("details"):
                details = step["details"]
                if isinstance(details, dict):
                    title_text = details.get("title", "")
                    body_text = strip_html(details.get("body", ""))
                    if title_text:
                        self.labeled_block(
                            f"Technical Detail: {title_text}", body_text, AMBER
                        )
                elif isinstance(details, list):
                    for d in details:
                        if isinstance(d, dict):
                            self.labeled_block(
                                f"Technical Detail: {d.get('title', '')}",
                                strip_html(d.get("body", "")),
                                AMBER,
                            )

            # Source code reference
            if step.get("source_function"):
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(*SLATE_400)
                ref = f"Source: {step.get('source_file', '')}:{step.get('source_line', '')} -- {step['source_function']}()"
                self.cell(0, 4, ref, new_x="LMARGIN", new_y="NEXT")
                self.ln(3)
            else:
                self.ln(2)


def build_title_page(pdf: PipelineDoc):
    """Create the title / cover page."""
    pdf.add_page()

    # Push content down
    pdf.ln(50)

    # Title
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 14, "Data Pipeline Documentation", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*INDIGO)
    pdf.cell(
        0,
        8,
        "GSTV Site Scoring Platform",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*SLATE_700)
    pdf.cell(
        0,
        6,
        "Machine Learning Pipeline for Advertising Site Revenue Prediction",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        6,
        "Covering 57,675+ Gas Station and Convenience Store Locations",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # Divider
    pdf.ln(12)
    pdf.set_draw_color(*INDIGO)
    pdf.set_line_width(0.5)
    center_x = pdf.w / 2
    pdf.line(center_x - 40, pdf.get_y(), center_x + 40, pdf.get_y())

    pdf.ln(12)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*SLATE_400)
    pdf.cell(0, 5, "Version 1.0", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Internal Use Only", align="C", new_x="LMARGIN", new_y="NEXT")


def build_executive_summary(pdf: PipelineDoc):
    """Add the executive summary page."""
    pdf.add_page()
    pdf.section_title("Executive Summary")
    pdf.body_text(
        "This document provides a comprehensive technical reference for the GSTV Site Scoring "
        "data pipeline. The pipeline ingests raw site data from multiple sources, transforms and "
        "cleans it through a series of ETL stages, trains machine learning models to predict "
        "advertising revenue potential, validates model performance, and deploys predictions "
        "as a production scoring system."
    )
    pdf.body_text(
        "The system processes data from approximately 57,675 unique gas station and convenience "
        "store locations, utilizing 18 source CSV files encompassing site attributes, revenue "
        "metrics, geographic proximity measurements, and retailer reference data. The pipeline "
        "supports two model architectures -- a PyTorch neural network with residual connections "
        "and XGBoost gradient-boosted trees -- both producing revenue predictions and lookalike "
        "classification scores."
    )

    pdf.subsection_title("Pipeline Overview")
    stages = [
        ("1. Data Collection", "Ingest 18 CSV source files from Salesforce and geodata pipelines"),
        ("2. Data Cleaning", "Standardize formats, handle nulls, encode flags, apply log transforms"),
        ("3. Data Combining", "Join temporal, momentum, and geospatial features into unified dataset"),
        ("4. Modeling", "Train neural network or XGBoost models for revenue prediction"),
        ("5. Testing & Validation", "Evaluate on held-out test set with unbiased metrics and SHAP analysis"),
        ("6. Productionizing", "Deploy as batch scoring API with filtered prediction and export capabilities"),
    ]
    for title, desc in stages:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*NAVY)
        pdf.cell(50, 6, title)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE_700)
        pdf.cell(0, 6, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.subsection_title("Key Metrics")
    metrics = [
        ("Total Sites Processed", "57,675"),
        ("Training Dataset Size", "26,096 active sites"),
        ("Source Features", "56 (16 numeric + 7 categorical + 33 boolean)"),
        ("Data Split", "70% train / 15% validation / 15% test"),
        ("Model Architectures", "PyTorch Neural Network, XGBoost"),
        ("Maximum Stored Experiments", "10 (FIFO rotation)"),
        ("Batch Inference Size", "4,096 sites per batch"),
        ("Export Formats", "CSV, XLSX"),
    ]
    for label, val in metrics:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE_700)
        pdf.cell(55, 5.5, label)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 5.5, val, new_x="LMARGIN", new_y="NEXT")


def build_table_of_contents(pdf: PipelineDoc, entries: list[dict]):
    """Add a table of contents page."""
    pdf.add_page()
    pdf.section_title("Table of Contents")
    pdf.ln(4)

    toc_items = [
        ("Executive Summary", 2),
        ("Table of Contents", 3),
    ]
    for i, entry in enumerate(entries):
        toc_items.append((entry["title"], 4 + i))  # approximate page numbers

    toc_items.append(("Appendix A: Technical Reference", 4 + len(entries)))
    toc_items.append(("Appendix B: Glossary of Terms", 5 + len(entries)))

    for title, page in toc_items:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*NAVY)
        # Title on left, dots, page number on right
        title_w = pdf.get_string_width(title) + 2
        pdf.cell(title_w, 7, title)
        dots_w = (pdf.w - 50) - title_w - 10
        dot_count = int(dots_w / pdf.get_string_width("."))
        pdf.set_text_color(*SLATE_400)
        pdf.cell(dots_w, 7, "." * max(dot_count, 3))
        pdf.set_text_color(*NAVY)
        pdf.cell(10, 7, str(page), align="R", new_x="LMARGIN", new_y="NEXT")


def build_stage_section(pdf: PipelineDoc, entry: dict):
    """Build a full section for one pipeline stage."""
    pdf.add_page()

    # Stage number and title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*INDIGO)
    pdf.cell(0, 12, entry["title"], new_x="LMARGIN", new_y="NEXT")

    # Guiding question
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(*SLATE_700)
    pdf.cell(0, 7, entry.get("question", ""), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Key statistics
    if entry.get("stats_row"):
        pdf.stat_row(entry["stats_row"])

    # Overview / introduction
    pdf.subsection_title("Overview")
    pdf.body_text(strip_html(entry.get("intro", "")))

    # Design rationale
    if entry.get("why"):
        pdf.labeled_block("Design Rationale", strip_html(entry["why"]))

    # Data sources
    if entry.get("sources"):
        pdf.source_table(entry["sources"])

    # Process steps
    if entry.get("steps"):
        pdf.step_list(entry["steps"])


def build_appendix_tech_ref(pdf: PipelineDoc, entries: list[dict]):
    """Appendix A: Technical Reference -- code locations."""
    pdf.add_page()
    pdf.section_title("Appendix A: Technical Reference")
    pdf.body_text(
        "The following table maps pipeline operations to their source code locations "
        "within the repository. All file paths are relative to the project root."
    )

    # Collect all code references
    refs = []
    for entry in entries:
        for step in entry.get("steps", []):
            if step.get("source_function"):
                refs.append(
                    {
                        "stage": entry["title"].split(". ", 1)[-1] if ". " in entry["title"] else entry["title"],
                        "step": strip_html(step.get("title", "")),
                        "file": step.get("source_file", ""),
                        "line": step.get("source_line", ""),
                        "function": step.get("source_function", ""),
                    }
                )

    if refs:
        col_widths = [30, 35, 45, 12, 35]
        headers = ["Stage", "Step", "File", "Line", "Function"]

        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*SLATE_400)
        pdf.set_fill_color(*LIGHT_BG)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 6, h, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*NAVY)
        for ref in refs:
            pdf.cell(col_widths[0], 5, ref["stage"][:18])
            pdf.cell(col_widths[1], 5, ref["step"][:22])
            pdf.cell(col_widths[2], 5, ref["file"])
            pdf.cell(col_widths[3], 5, str(ref["line"]))
            pdf.cell(col_widths[4], 5, ref["function"] + "()")
            pdf.ln()


def build_appendix_glossary(pdf: PipelineDoc):
    """Appendix B: Glossary of common terms."""
    pdf.add_page()
    pdf.section_title("Appendix B: Glossary of Terms")

    terms = [
        ("Active Site", "A gas station or convenience store that is currently operational and eligible for advertising placement. Only active sites are included in model training."),
        ("BatchNorm", "Batch Normalization. A neural network technique that normalizes layer inputs to stabilize and accelerate training."),
        ("BCEWithLogitsLoss", "Binary Cross-Entropy loss with built-in sigmoid. Used for classification training, with pos_weight to handle class imbalance."),
        ("DMA", "Designated Market Area. A geographic region used in media planning to define television markets."),
        ("Early Stopping", "A training technique that halts model optimization when validation performance stops improving, preventing overfitting."),
        ("Experiment Folder", "A self-contained directory (job_xxx/) storing all artifacts from a single training run: config, model weights, preprocessor state, and evaluation results."),
        ("Feature Processing", "The transformation of raw data columns into model-ready inputs via scaling, encoding, and type conversion."),
        ("GTVID", "A unique geographic identifier assigned to each site location."),
        ("Gbase ID (id_gbase)", "An alternate unique identifier for sites, used in the Salesforce data exports."),
        ("Haversine Formula", "A trigonometric formula for computing great-circle distances between two latitude/longitude points on Earth."),
        ("HuberLoss", "A loss function that is quadratic for small errors and linear for large errors, making it robust to outliers in revenue data."),
        ("Left Join", "A database join that preserves all rows from the left (primary) table, adding matched columns from the right table or null values where no match exists."),
        ("Lookalike Classification", "A binary classification task that identifies sites resembling top performers. Sites above a configurable percentile threshold are labeled as positive."),
        ("MPS", "Metal Performance Shaders. Apple Silicon's GPU acceleration framework, used by PyTorch for faster neural network training on Mac hardware."),
        ("Parquet", "A columnar file format optimized for analytical workloads. Supports efficient compression and fast column-level reads."),
        ("Polars", "A high-performance DataFrame library written in Rust, used in the ETL pipeline for 10-20x faster processing than pandas."),
        ("pos_weight", "A training parameter that upweights the minority class in binary classification to compensate for class imbalance (e.g., 9.0 means positive examples count 9x more)."),
        ("Preprocessor", "The fitted collection of scalers, encoders, and vocabulary mappings saved as preprocessor.pkl. Required for consistent feature transformation during inference."),
        ("Relative Strength (RS)", "A momentum indicator comparing recent performance to historical averages. RS > 1.0 indicates an upward trend; RS < 1.0 indicates decline."),
        ("Residual Connection", "A neural network architecture pattern where the input to a layer is added to its output, enabling deeper networks by ensuring gradient flow."),
        ("SHAP", "SHapley Additive exPlanations. A game-theory-based method for explaining individual predictions by attributing contributions to each feature."),
        ("SSE", "Server-Sent Events. A unidirectional HTTP streaming protocol used to push real-time training progress updates to the browser UI."),
        ("StandardScaler", "A scikit-learn transformer that centers features to zero mean and unit variance. Fitted on training data and applied identically during inference."),
        ("XGBoost", "Extreme Gradient Boosting. An ensemble learning method that builds sequential decision trees, each correcting the errors of the previous one."),
    ]

    for term, definition in terms:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, term, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*SLATE_700)
        pdf.multi_cell(0, 4.5, definition)
        pdf.ln(2)


def main():
    entries = load_stage_data()
    pdf = PipelineDoc()

    # Build document structure
    build_title_page(pdf)
    build_executive_summary(pdf)
    build_table_of_contents(pdf, entries)

    for entry in entries:
        build_stage_section(pdf, entry)

    build_appendix_tech_ref(pdf, entries)
    build_appendix_glossary(pdf)

    # Write PDF
    pdf.output(str(OUTPUT_PATH))
    print(f"Generated: {OUTPUT_PATH}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    main()
