"""
Lets mockup the data readed as a pandas.Dataframe
"""

import pandas as pd

RATINGS = "sr/data/user_item_2009_06.csv"
ARTIST = "sr/data/artist2.csv"

# Read data as a pandas dataframe
def load_data(path: str, sep: str):
    """
    Loads data from csv file

    Parameters
    ----------
    path : str
        Path to the csv file
    sep : str
        Column separator

    Returns
    -------
    pd.DataFrame
        Loaded dataframe from csv
    """
    data = pd.read_csv(path, sep=sep)
    return data


def parse_neighbors_name(neighbors: list, artists: pd.DataFrame):
    """
    Parse the raw id from the ratings to the artist name

    Parameters
    ----------
    neighbors : list
        A list with tuples containing the neighbors raw id and its estimation
    artists : pd.DataFrame
        Artist information dataframe

    Returns
    -------
    list:
        A list with tuples containing the neighbors name and its estimation
    """
    neighbors_df = pd.DataFrame(data=neighbors, columns=["item", "estimation"])
    subset = pd.merge(neighbors_df, artists, left_on="item", right_on="artist_id")
    subset = subset[["artist_name", "estimation"]]
    return subset.to_numpy()