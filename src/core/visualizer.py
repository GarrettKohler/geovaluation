"""
Map visualization for geospatial datasets.

This module provides flexible map rendering that can display
arbitrary datasets with configurable styling and colormaps.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors


class ColorScale(Enum):
    """Pre-defined color scales for map visualization."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    BLUES = "Blues"
    GREENS = "Greens"
    REDS = "Reds"
    ORANGES = "Oranges"
    PURPLES = "Purples"
    GREYS = "Greys"
    YELLOW_GREEN_BLUE = "YlGnBu"
    YELLOW_ORANGE_RED = "YlOrRd"
    RED_YELLOW_GREEN = "RdYlGn"
    SPECTRAL = "Spectral"
    COOLWARM = "coolwarm"


@dataclass
class MapStyle:
    """Configuration for map styling.

    Attributes:
        colormap: Color scale to use for values
        edge_color: Color for polygon boundaries
        edge_width: Width of polygon boundaries
        alpha: Transparency (0-1)
        missing_color: Color for regions with no data
        figsize: Figure size in inches (width, height)
        title: Map title
        legend: Whether to show legend/colorbar
        legend_label: Label for the colorbar
    """
    colormap: Union[str, ColorScale] = ColorScale.VIRIDIS
    edge_color: str = "black"
    edge_width: float = 0.5
    alpha: float = 1.0
    missing_color: str = "lightgrey"
    figsize: Tuple[int, int] = (12, 8)
    title: Optional[str] = None
    legend: bool = True
    legend_label: Optional[str] = None

    def get_colormap_name(self) -> str:
        """Get the colormap name as a string."""
        if isinstance(self.colormap, ColorScale):
            return self.colormap.value
        return self.colormap


@dataclass
class LayerConfig:
    """Configuration for a map layer.

    Attributes:
        data: GeoDataFrame to display
        value_column: Column containing values to visualize (None for uniform color)
        style: Styling options for this layer
        label: Label for this layer in legend
        zorder: Drawing order (higher = on top)
    """
    data: gpd.GeoDataFrame
    value_column: Optional[str] = None
    style: MapStyle = field(default_factory=MapStyle)
    label: Optional[str] = None
    zorder: int = 1


class MapVisualizer:
    """Creates map visualizations from geospatial data.

    This class provides a flexible interface for creating choropleth maps
    and multi-layer visualizations from arbitrary GeoDataFrames.
    """

    def __init__(self, style: Optional[MapStyle] = None):
        """Initialize the visualizer.

        Args:
            style: Default style settings for maps
        """
        self.default_style = style or MapStyle()
        self._layers: List[LayerConfig] = []

    def add_layer(
        self,
        data: gpd.GeoDataFrame,
        value_column: Optional[str] = None,
        style: Optional[MapStyle] = None,
        label: Optional[str] = None,
        zorder: int = 1
    ) -> "MapVisualizer":
        """Add a layer to the visualization.

        Args:
            data: GeoDataFrame to display
            value_column: Column containing values to visualize
            style: Styling options (uses default if not specified)
            label: Label for legend
            zorder: Drawing order

        Returns:
            Self for method chaining
        """
        layer = LayerConfig(
            data=data,
            value_column=value_column,
            style=style or self.default_style,
            label=label,
            zorder=zorder
        )
        self._layers.append(layer)
        return self

    def clear_layers(self) -> "MapVisualizer":
        """Remove all layers.

        Returns:
            Self for method chaining
        """
        self._layers = []
        return self

    def plot(
        self,
        data: Optional[gpd.GeoDataFrame] = None,
        value_column: Optional[str] = None,
        style: Optional[MapStyle] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """Create a map visualization.

        If data is provided, creates a single-layer map.
        If no data is provided, renders all added layers.

        Args:
            data: GeoDataFrame to visualize (optional if layers added)
            value_column: Column containing values to visualize
            style: Styling options
            ax: Existing axes to plot on (creates new if None)

        Returns:
            Tuple of (Figure, Axes)
        """
        style = style or self.default_style

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=style.figsize)
        else:
            fig = ax.get_figure()

        # If data provided directly, create single layer plot
        if data is not None:
            self._plot_layer(
                LayerConfig(data=data, value_column=value_column, style=style),
                ax,
                show_colorbar=style.legend
            )
        else:
            # Plot all added layers
            for i, layer in enumerate(sorted(self._layers, key=lambda x: x.zorder)):
                show_colorbar = (
                    layer.style.legend and
                    layer.value_column is not None and
                    i == len(self._layers) - 1  # Only last layer gets colorbar
                )
                self._plot_layer(layer, ax, show_colorbar=show_colorbar)

        # Set title
        if style.title:
            ax.set_title(style.title, fontsize=14, fontweight='bold')

        # Clean up axes
        ax.set_axis_off()

        plt.tight_layout()
        return fig, ax

    def _plot_layer(
        self,
        layer: LayerConfig,
        ax: Axes,
        show_colorbar: bool = True
    ):
        """Plot a single layer on the axes."""
        style = layer.style
        cmap = style.get_colormap_name()

        plot_kwargs: Dict[str, Any] = {
            "ax": ax,
            "edgecolor": style.edge_color,
            "linewidth": style.edge_width,
            "alpha": style.alpha,
            "zorder": layer.zorder
        }

        if layer.value_column is not None:
            # Choropleth map
            plot_kwargs.update({
                "column": layer.value_column,
                "cmap": cmap,
                "legend": show_colorbar,
                "missing_kwds": {"color": style.missing_color}
            })

            if show_colorbar and style.legend_label:
                plot_kwargs["legend_kwds"] = {"label": style.legend_label}

        else:
            # Uniform color
            plot_kwargs["color"] = style.missing_color

        layer.data.plot(**plot_kwargs)

    def choropleth(
        self,
        data: gpd.GeoDataFrame,
        value_column: str,
        title: Optional[str] = None,
        colormap: Union[str, ColorScale] = ColorScale.VIRIDIS,
        legend_label: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Tuple[Figure, Axes]:
        """Create a choropleth map with minimal configuration.

        This is a convenience method for quickly creating standard
        choropleth visualizations.

        Args:
            data: GeoDataFrame with values to visualize
            value_column: Column containing numeric values
            title: Map title
            colormap: Color scale to use
            legend_label: Label for the colorbar
            figsize: Figure size in inches

        Returns:
            Tuple of (Figure, Axes)

        Example:
            >>> viz = MapVisualizer()
            >>> fig, ax = viz.choropleth(
            ...     data=zip_walkability,
            ...     value_column="NatWalkInd_mean",
            ...     title="Walkability by ZIP Code",
            ...     colormap=ColorScale.YELLOW_GREEN_BLUE
            ... )
        """
        style = MapStyle(
            colormap=colormap,
            title=title,
            legend_label=legend_label or value_column,
            figsize=figsize
        )
        return self.plot(data, value_column, style)

    def save(
        self,
        filepath: Union[str, Path],
        fig: Optional[Figure] = None,
        dpi: int = 150,
        **kwargs
    ):
        """Save the visualization to a file.

        Args:
            filepath: Output file path
            fig: Figure to save (uses current figure if None)
            dpi: Resolution in dots per inch
            **kwargs: Additional arguments passed to savefig
        """
        if fig is None:
            fig = plt.gcf()

        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches='tight',
            **kwargs
        )


def plot_map(
    data: gpd.GeoDataFrame,
    value_column: str,
    title: Optional[str] = None,
    colormap: str = "viridis",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Convenience function to quickly plot a choropleth map.

    Args:
        data: GeoDataFrame with values to visualize
        value_column: Column containing numeric values
        title: Map title
        colormap: Matplotlib colormap name
        figsize: Figure size in inches
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure, Axes)

    Example:
        >>> fig, ax = plot_map(
        ...     data=walkability_by_zip,
        ...     value_column="NatWalkInd_mean",
        ...     title="Walkability Index by ZIP Code"
        ... )
    """
    viz = MapVisualizer()
    fig, ax = viz.choropleth(
        data=data,
        value_column=value_column,
        title=title,
        colormap=colormap,
        figsize=figsize
    )

    if save_path:
        viz.save(save_path, fig)

    return fig, ax


def compare_maps(
    datasets: List[Tuple[gpd.GeoDataFrame, str, str]],
    ncols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    colormap: str = "viridis"
) -> Tuple[Figure, List[Axes]]:
    """Create a grid of maps for comparison.

    Args:
        datasets: List of (GeoDataFrame, value_column, title) tuples
        ncols: Number of columns in the grid
        figsize: Figure size (auto-calculated if None)
        colormap: Colormap to use for all maps

    Returns:
        Tuple of (Figure, list of Axes)

    Example:
        >>> datasets = [
        ...     (income_data, "median_income", "Median Income"),
        ...     (walkability_data, "walk_score", "Walk Score"),
        ...     (crime_data, "crime_rate", "Crime Rate"),
        ... ]
        >>> fig, axes = compare_maps(datasets, ncols=2)
    """
    n = len(datasets)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes array for easy iteration
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    viz = MapVisualizer()

    for i, (data, value_col, title) in enumerate(datasets):
        style = MapStyle(
            colormap=colormap,
            title=title,
            legend_label=value_col,
            figsize=figsize
        )
        viz.plot(data, value_col, style, ax=axes[i])

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes[:n]
