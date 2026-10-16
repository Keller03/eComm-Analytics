import pandas as pd

def loadData(source):
    df = pd.read_csv(source)
    return df

