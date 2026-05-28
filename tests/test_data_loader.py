import os
import pandas as pd
from src import data_loader


def test_safe_read_csv_and_validate(tmp_path):
    # Create a small CSV
    csv_path = tmp_path / "mini.csv"
    df = pd.DataFrame({"Resume_str": ["sample resume text"], "Category": ["TEST"]})
    df.to_csv(csv_path, index=False)

    # Read via safe_read_csv
    read = data_loader.safe_read_csv(str(csv_path))
    assert read is not None
    assert "Resume_str" in read.columns

    # Validate schema
    ok, msg = data_loader.validate_schema(read, ["Resume_str"])
    assert ok
    assert msg == "OK"
