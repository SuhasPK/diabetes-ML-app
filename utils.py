# utils.py

import pandas as pd

def load_data(file_path):
    """Load CSV data."""
    return pd.read_csv(file_path)

def save_figure(fig, filename):
    """Save Plotly figure as an image."""
    fig.write_image(filename)
