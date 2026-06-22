"""Ingestion-side settings: extraction, parent/child split, reconcile, infra.

Grouped under ``IngestionConfig`` and folded into ``AppConfig`` — read via
``app_config`` (e.g. ``app_config.BUCKET_NAME``, ``app_config.PC_PARENT_MAX_CHARS``).
No standalone instance, no module constants. Domain enums live in
``ingestion/constants.py``.
"""
from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class ExtractConfig(BaseSettings):
    """3A tunables: excerpt budgets + hybrid PDF extraction knobs."""

    CLASSIFY_MAX_CHARS: PositiveInt = Field(default=6000, description="Leading chars fed to the domain classifier.")
    METADATA_MAX_CHARS: PositiveInt = Field(default=8000, description="Leading chars fed to the metadata extractor.")
    PDF_RENDER_SCALE: PositiveFloat = Field(default=2.0, description="Page render scale (~2.0 ≈ 144 dpi).")
    PDF_VISION_MAX_PAGES: NonNegativeInt = Field(default=3, description="A PDF with at most this many pages is extracted WHOLESALE by the vision model (every page rendered & read by the VLM) instead of the text-layer hybrid path — short docs are disproportionately scans/forms/figure-heavy pages where vision is more faithful. 0 disables (always use the hybrid path).")
    PDF_SCAN_MIN_CHARS: NonNegativeInt = Field(default=80, description="Page text below this ⇒ scanned → VLM OCR.")
    PDF_HEADING_MIN_DELTA: PositiveFloat = Field(default=1.0, description="Font points above body size ⇒ heading.")
    PDF_HEADING_MAX_CHARS: PositiveInt = Field(default=120, description="A heading line is short.")
    PDF_CHART_MIN_CURVES: PositiveInt = Field(default=40, description="Vector primitives ⇒ render page as figure.")
    PDF_TABLE_TEXT_FALLBACK: bool = Field(default=True, description="When lines-based table detection finds none on a page, retry with the text-alignment strategy to LOCATE borderless tables (common in contracts/invoices).")
    PDF_TABLE_MIN_ROWS: PositiveInt = Field(default=2, description="Min rows for a text-detected (borderless) table region to be accepted (false-positive filter).")
    PDF_TABLE_MIN_COLS: PositiveInt = Field(default=2, description="Min columns for a text-detected (borderless) table region to be accepted (false-positive filter).")
    PDF_TABLE_VLM_FALLBACK: bool = Field(default=True, description="For a SUSPECT table (row-collapsed or sparse — any detection strategy), crop its region to PNG and transcribe via the vision model instead of trusting pdfplumber's cell extraction. Disable to stay pdfplumber-only (cheaper, less accurate on broken grids).")
    PDF_TABLE_VLM_MAX_FILL: PositiveFloat = Field(default=0.7, description="Cell-fill ratio gate: a detected table below this is 'sparse' (columns/rows mis-segmented) → render→VLM; at/above it pdfplumber's extraction is trusted (unless row-collapsed).")
    PDF_TABLE_MAX_CELL_LINES: PositiveInt = Field(default=3, description="A table cell holding this many+ internal line breaks ⇒ several source rows were merged into one (row-collapse) ⇒ the table is suspect → render→VLM.")


class SplitConfig(BaseSettings):
    """3B chunking (CHARACTER-based) + level3 batching.

    NOTE: the cross-field relationships (overlap ≤ child ≤ parent,
    parent_min ≤ parent_max) are validated at use-time in
    ``ingestion/splitters/parent_child.py:_validate_thresholds`` — not here.
    """

    PC_PARENT_MAX_CHARS: PositiveInt = Field(default=10000, description="Parent ceiling — split parents (text or table) larger than this.")
    PC_PARENT_MIN_CHARS: PositiveInt = Field(default=2000, description="Parent floor — merge/fold parents smaller than this into a neighbour.")
    PC_CHILD_MIN_CHARS: PositiveInt = Field(default=300, description="Minimum child chunk size.")
    PC_CHILD_MAX_CHARS: PositiveInt = Field(default=512, description="Maximum child chunk size.")
    PC_CHILD_OVERLAP: NonNegativeInt = Field(default=40, description="Child sliding-window overlap.")
    LEVEL3_BATCH_MAX_CHARS: PositiveInt = Field(default=12000, description="Prompt-size budget per level3 LLM call.")
    LEVEL3_MAX_PARENTS_PER_CALL: PositiveInt = Field(default=12, description="Max parents per level3 LLM call.")


class ReconcileConfig(BaseSettings):
    """Reconciler tunables."""

    RECONCILE_INTERVAL: PositiveInt = Field(default=300, description="Beat tick cadence (seconds).")
    RECONCILE_STUCK_AFTER: PositiveInt = Field(default=900, description="A doc idle longer than this is 'stuck'.")
    MAX_ATTEMPTS: PositiveInt = Field(default=5, description="Enqueue attempts before dead-lettering.")


class InfraConfig(BaseSettings):
    """Deployed-resource identifiers read from ``.env`` (overridable in tests via
    ``monkeypatch.setattr(app_config, ...)``). S3/storage settings now live in
    ``config.storage_config.StorageConfig``."""

    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Celery broker / Redis URL.")


class IngestionConfig(ExtractConfig, SplitConfig, ReconcileConfig, InfraConfig):
    """Umbrella over the ingestion sub-configs. Never instantiated alone —
    folded into ``AppConfig``."""
