#functions to load both datasets - Read paths and manages dtypes - returns dataframe VIA pandas 
import pandas as pd

def load_lending_club(path: str, nrows: int | None = 50_000) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        nrows=nrows,
        low_memory=False,
        dtype={"zip_code": str},
    )
    return df

def load_census_income(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"zip code tabulation area": str},
    )
    return df
