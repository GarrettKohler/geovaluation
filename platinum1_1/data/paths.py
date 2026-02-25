"""
Centralized, configurable path resolution for all data files.

Every file path the application needs is exposed as a property so that
downstream code never hard-codes directory layouts.  Paths are derived
from the application Settings, which can be overridden via environment
variables for testing or deployment.
"""

from pathlib import Path

from ..config.settings import get_settings, Settings


class DataPaths:
    """
    Centralized path resolution for input, processed, and auxiliary data.

    Usage::

        paths = DataPaths()               # uses default settings
        paths = DataPaths(custom_settings) # override for testing

        paths.site_scores_csv       # -> .../data/input/site_scores_revenue_and_diagnostics.csv
        paths.training_parquet      # -> .../data/processed/site_training_data.parquet
        paths.mcdonalds_geodata     # -> .../data/input/platinum/mcdonalds_geodata.csv
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -------------------------------------------------------------------------
    # Top-level directories
    # -------------------------------------------------------------------------

    @property
    def input_dir(self) -> Path:
        """Main data/input directory (contains the 927 MB CSV)."""
        return self._settings.DATA_INPUT_DIR

    @property
    def platinum_dir(self) -> Path:
        """Auxiliary geodata directory (platinum-specific CSVs)."""
        return self._settings.DATA_PLATINUM_DIR

    @property
    def processed_dir(self) -> Path:
        """Pre-processed parquet directory."""
        return self._settings.DATA_PROCESSED_DIR

    @property
    def output_dir(self) -> Path:
        """Model output directory (experiments, checkpoints)."""
        return self._settings.OUTPUT_DIR

    # -------------------------------------------------------------------------
    # Primary data files
    # -------------------------------------------------------------------------

    @property
    def site_scores_csv(self) -> Path:
        """Raw monthly site scores (927 MB, ~1.4 M rows)."""
        return self.input_dir / "site_scores_revenue_and_diagnostics.csv"

    @property
    def training_parquet(self) -> Path:
        """Pre-processed training data (one row per site, ML-ready)."""
        return self.processed_dir / "site_training_data.parquet"

    @property
    def precleaned_parquet(self) -> Path:
        """Intermediate aggregated parquet (all sites, pre-cleaned)."""
        return self.processed_dir / "site_aggregated_precleaned.parquet"

    # -------------------------------------------------------------------------
    # Platinum auxiliary files (geodata and distances)
    # -------------------------------------------------------------------------

    @property
    def nearest_site_distances(self) -> Path:
        """Pre-computed nearest-site distances."""
        return self.platinum_dir / "nearest_site_distances.csv"

    @property
    def interstate_distances(self) -> Path:
        """Pre-computed interstate distances per site."""
        return self.platinum_dir / "site_interstate_distances.csv"

    @property
    def mcdonalds_geodata(self) -> Path:
        """Raw McDonald's locations (lat/lon)."""
        return self.platinum_dir / "mcdonalds_geodata.csv"

    @property
    def walmart_geodata(self) -> Path:
        """Raw Walmart locations (lat/lon)."""
        return self.platinum_dir / "walmart_geodata.csv"

    @property
    def target_geodata(self) -> Path:
        """Raw Target locations (lat/lon)."""
        return self.platinum_dir / "target_geo_data.csv"

    @property
    def salesforce_revenue(self) -> Path:
        """Salesforce site revenue export."""
        return self.platinum_dir / "salesforce_site_revenue.csv"

    @property
    def sites_base_data(self) -> Path:
        """Sites base data set (restrictions, capabilities, demographics)."""
        return self.platinum_dir / "sites_base_data_set.csv"

    # -------------------------------------------------------------------------
    # Convenience: existence checks
    # -------------------------------------------------------------------------

    def validate(self) -> dict[str, bool]:
        """
        Return a dict of ``{label: exists}`` for every known path.

        Useful for startup diagnostics and health checks.
        """
        paths = {
            "site_scores_csv": self.site_scores_csv,
            "training_parquet": self.training_parquet,
            "precleaned_parquet": self.precleaned_parquet,
            "nearest_site_distances": self.nearest_site_distances,
            "interstate_distances": self.interstate_distances,
            "mcdonalds_geodata": self.mcdonalds_geodata,
            "walmart_geodata": self.walmart_geodata,
            "target_geodata": self.target_geodata,
            "salesforce_revenue": self.salesforce_revenue,
            "sites_base_data": self.sites_base_data,
        }
        return {label: p.exists() for label, p in paths.items()}
