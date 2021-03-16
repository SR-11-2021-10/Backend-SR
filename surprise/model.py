"""
In this module, we set all logic required to implement the recommender system
"""

import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import KNNBaseline
from surprise import accuracy
from surprise.model_selection import train_test_split

# Let's write some constants
USER_USER = "USER_USER"
ITEM_ITEM = "ITEM_ITEM"
JACCARD = "JACCARD"
COSINE = "cosine"
PEARSON = "pearson_baseline"

# Type of model: user-user, item-item
model_type = {
    USER_USER: USER_USER,
    ITEM_ITEM: ITEM_ITEM,
    JACCARD: JACCARD,
    COSINE: COSINE,
    PEARSON: PEARSON,
}


class RecommenderSystem:
    """
    This class defines a recommender system using Surprise framework
    Parameters required are the type of the model, the similtude measure and
    finally the user-item rating dataframe
    """

    def __init__(self, type: str, similitude: str, user_item_rating: pd.DataFrame):
        """
        Creates a new recommender system to train a model and predict
        recommendations

        Parameters
        ----------
        type : str
            Create a new memory-based model (User-User) or model-based model (Item-Item)
        similitud : str
            Similitud measure to use: JACCARD, COSINE, PEARSON
        user_item_rating : pd.DataFrame
            Dataframe including all users rating to an item

        Attributes
        ----------
        type : str
            Create a new memory-based model (User-User) or model-based model (Item-Item)
        similitud : str
            Similitud measure to use: JACCARD, COSINE, PEARSON
        user_item_rating : pd.DataFrame
            Dataframe including all users rating to an item
        """
        self.type = type
        self.similitude = similitude
        self.user_item_rating = user_item_rating

        # Load data
        self.data = self.__load_data()

        # Create model
        self.model = self.__instantiate_model()

        # Fits the model
        self.__fit()

    def __load_data(self):
        """
        Loads data from pandas dataframe. The dataframe must be structured as
        user, item, rating

        Returns
        -------
        surprise.Dataset:
            Dataset loaded to be used with surprise
        """
        reader = Reader(rating_scale=(1, 5))
        return Dataset.load_from_df(self.user_item_rating, reader)

    def __instantiate_model(self):
        """
        Instantiates an specific model according to the parameters received.

        Returns
        -------
        surprise.KNNBaseline:
            Model to be trained and used to make predictions
        """

        # User-User Model
        if self.type == USER_USER:
            # Cosine distance
            if self.similitude == COSINE:
                return KNNBaseline(sim_options={"name": COSINE, "user_base": True})
            elif self.similitude == PEARSON:
                return KNNBaseline(sim_options={"name": PEARSON, "user_base": True})
            elif self.similitude == JACCARD:
                pass
            else:
                raise RuntimeError("[RecommenderSystem] Similitude measure not found")
            pass
        # Item-Item Model
        elif self.type == ITEM_ITEM:
            # Cosine distance
            if self.similitude == COSINE:
                return KNNBaseline(sim_options={"name": COSINE, "user_base": False})
            elif self.similitude == PEARSON:
                return KNNBaseline(sim_options={"name": PEARSON, "user_base": False})
            elif self.similitude == JACCARD:
                pass
            else:
                raise RuntimeError("[RecommenderSystem] Similitude measure not found")
            pass
        else:
            raise RuntimeError(
                "[RecommenderSystem] You can only set a user-user or item-item model"
            )

    def __fit(self):
        """
        Fits the model with 80% of all available data
        """
        train, test = train_test_split(self.data, test_size=0.2)
        self.model.fit(train)

        predictions = self.model.test(test)
        rmse = accuracy.rmse(predictions)
        print(f"Accuracy: {rmse}")

    def predict(self, uid: str, iid: str) -> dict:
        """
        Makes a new prediction for an unseen item according to the model created

        Parameters
        ----------
        uid : str
            User id
        iid: str
            Item id

        Returns
        -------
        dict:
            Information about prediction: Estimated and list of neighboors
        """

        pred = self.model.predict(uid=uid, iid=iid)
        # Lets retrieve 5 neighboors
        item_neighboors = self.model.get_neighbors(iid, k=5)
        item_id_neighbors = [
            self.model.trainset.to_raw_iid(inner_id) for inner_id in item_neighboors
        ]

        return {
            "user": uid,
            "item": iid,
            "estimation": pred[3],
            "neighbors": item_id_neighbors,
        }
