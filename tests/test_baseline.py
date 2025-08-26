from pathlib import Path
import pandas as pd
from src.models.baseline import naive_last_value

def test_naive_shape(tmp_path):
    # Minimal sanity: create tiny frame and ensure output aligns
    df = pd.DataFrame({
        "Close": [1.0, 1.1, 1.2],
        "SMA_5": [1.0, 1.05, 1.1],
        "SMA_20": [1.0, 1.02, 1.04],
        "Target_Close_t+1": [1.1, 1.2, 1.3]
    })
    pred = naive_last_value(df)
    assert len(pred) == len(df)
