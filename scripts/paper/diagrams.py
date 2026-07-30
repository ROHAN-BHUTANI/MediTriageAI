import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

logger = logging.getLogger(__name__)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str = "lightblue",
) -> None:
    """Draw a styled box with text on a matplotlib axes.

    Args:
        ax (plt.Axes): The axes to draw on.
        xy (Tuple[float, float]): The bottom-left coordinate of the box.
        width (float): Box width.
        height (float): Box height.
        text (str): The text to display inside the box.
        facecolor (str, optional): The background color of the box. Defaults to 'lightblue'.
    """
    rect = patches.Rectangle(
        xy, width, height, linewidth=1, edgecolor="black", facecolor=facecolor
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
    )


def generate_architecture_diagram(output_path: Path) -> None:
    """Generate the E-PATH-CO-REASON architecture diagram.

    Args:
        output_path (Path): Path to save the diagram (supports .pdf, .svg, .png).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    draw_box(ax, (3, 8), 4, 1.5, "Raw Clinical Input", facecolor="#e0f7fa")
    draw_box(
        ax, (3, 5), 4, 1.5, "E-PATH-CO-REASON\\n(XLM-RoBERTa)", facecolor="#c8e6c9"
    )
    draw_box(
        ax, (1, 2), 3, 1.5, "Specialist Routing\\n(Department)", facecolor="#ffccbc"
    )
    draw_box(ax, (6, 2), 3, 1.5, "Triage Severity\\n(Level S1-S5)", facecolor="#ffccbc")

    # Arrows
    ax.annotate(
        "", xy=(5, 6.5), xytext=(5, 8), arrowprops=dict(arrowstyle="->", lw=1.5)
    )
    ax.annotate(
        "", xy=(2.5, 3.5), xytext=(4, 5), arrowprops=dict(arrowstyle="->", lw=1.5)
    )
    ax.annotate(
        "", xy=(7.5, 3.5), xytext=(6, 5), arrowprops=dict(arrowstyle="->", lw=1.5)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated architecture diagram at {output_path}")


def generate_pipeline_diagram(output_path: Path) -> None:
    """Generate the data builder pipeline diagram.

    Args:
        output_path (Path): Path to save the diagram (supports .pdf, .svg, .png).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        "Ingest",
        "Schema Align",
        "Normalize",
        "Deduplicate",
        "Filter",
        "Partitions",
        "Export",
    ]
    for i, stage in enumerate(stages):
        draw_box(ax, (i * 2, 1.5), 1.6, 1, stage, facecolor="#bbdefb")
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(i * 2 + 1.8, 2),
                xytext=(i * 2 + 1.6, 2),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated pipeline diagram at {output_path}")
