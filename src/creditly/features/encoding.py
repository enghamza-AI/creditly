
import pandas as pd


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  
    return pd.get_dummies(df, columns=columns, drop_first=True)
  


class TargetEncoder:
   
    def __init__(self, column: str, smoothing: float = 10.0):
        self.column = column
       
        self.smoothing = smoothing
        self.mapping_ = None
        self.global_mean_ = None

    def fit(self, df: pd.DataFrame, target: pd.Series):
        
        self.global_mean_ = target.mean()

        stats = target.groupby(df[self.column]).agg(["mean", "count"])

        
        smoothed = (
            stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
        ) / (stats["count"] + self.smoothing)

        self.mapping_ = smoothed.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        
        if self.mapping_ is None:
            raise RuntimeError("Call fit() before transform() — encoding must be learned from training data first.")
        return df[self.column].map(self.mapping_).fillna(self.global_mean_)
