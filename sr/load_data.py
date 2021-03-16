"""
Lets mockup the data readed as a pandas.Dataframe
"""

import pandas as pd

# Read data as a pandas dataframe
def load_data():
    data = pd.read_csv("sr/data/user_item_2009_06.csv", sep=",")
    return data