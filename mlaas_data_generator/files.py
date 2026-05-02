"""Functions for comining data files"""

import pandas as pd
from pathlib import Path
from typing import List, Union


def combine_data_files(
        paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        id_col: str = "Maas_ID",
        start_id: int = 1,
        dedupe: bool = False) -> pd.DataFrame:
    if not paths:
        raise ValueError("No input files provided")
    
    dfs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        df = pd.read_csv(p)
        dfs.append(df)

    if not dfs:
        raise ValueError("No CSVs where found in provided paths")
    
    combined_df = pd.concat(dfs, ignore_index=True)

    if dedupe:
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    combined_df[id_col] = range(start_id, start_id + len(combined_df))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    return combined_df 


    
