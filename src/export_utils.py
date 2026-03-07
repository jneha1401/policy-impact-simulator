import pandas as pd
import json
import os
from datetime import datetime

def export_to_csv(df, filename_prefix="policy_export"):
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"exports/{filename_prefix}_{timestamp}.csv"
    df.to_csv(filepath, index=False)
    return filepath

def export_brief_to_json(brief_text, metadata):
    os.makedirs("exports/briefs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"exports/briefs/brief_{timestamp}.json"
    payload = {"timestamp": timestamp, "metadata": metadata, "content": brief_text}
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=4)
    return filepath
