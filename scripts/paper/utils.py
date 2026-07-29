import logging
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def generate_placeholder(output_path: Path, text: str) -> None:
    """Generate a placeholder figure for missing experiments.

    Args:
        output_path (Path): Path to save the placeholder figure.
        text (str): Text to display on the placeholder.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.axis('off')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Generated placeholder at {output_path}")

def save_table(df: pd.DataFrame, output_base_path: Path) -> None:
    """Save a Pandas DataFrame as CSV, LaTeX, and Markdown.

    Args:
        df (pd.DataFrame): The table dataframe to save.
        output_base_path (Path): The base path without extensions (e.g., 'path/to/table').
    """
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base_path.with_suffix('.csv')
    tex_path = output_base_path.with_suffix('.tex')
    md_path = output_base_path.with_suffix('.md')
    
    df.to_csv(str(csv_path), index=False)
    try:
        df.to_latex(str(tex_path), index=False)
    except AttributeError:
        # Pandas > 2.0 uses styler for latex
        df.style.to_latex(str(tex_path))
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
        
    logger.info(f"Saved table to {output_base_path} (.csv, .tex, .md)")
