"""SQLite database storing information of interest for making bottom-up projections."""

import pandas as pd
import sqlite3
from pathlib import Path



def set_company_prod(company_prod_dir: Path, save_dir: Path):
    """Insert historical company production values for top 50 producers, from WSA.

    Args:
        company_prod_dir (Path): _description_
        save_dir (Path): _description_
    """
    dfs = [pd.read_excel(fpath) for fpath in company_prod_dir.glob("top_50_*.xlsx")]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    table_name = "steel_producers"
    conn = sqlite3.connect(save_dir / "steel_sector.db")
    df.to_sql("steel_producers", conn, if_exists="replace", index=False)


