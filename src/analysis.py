import pandas as pd

def inspect_data(df: pd.DataFrame) -> dict:
    return {
        "shape" : df.shape,
        "missing_values" : df.isna().sum(),
        "missing_percentage" : df.isna().mean() * 100,
        "duplicates" : df.duplicated().sum(),
        "unique_values" : df.nunique(),
        "statistics" : df.describe()
    }

def print_inspection(results: dict) -> None:
    for key, value in results.items():
        print(f"\n--- {key.upper()} ---")
        print(value)

def calculate_pearson_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")

    return numeric_df.corr()


