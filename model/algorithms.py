"""
Creates and instantiates all the recommender models
"""
import itertools
from collections import defaultdict
from math import log
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.sparse as sp
import surprise as spr
from main import dict_words_business, dict_words_user, log_regression


def create_svd_model() -> spr.SVD:
    """
    Creates an Single Value Decomposition model with the best metrics found
    in experimentation
    """
    return spr.SVD(
        n_factors=5,
        n_epochs=200,
        biased=True,
        lr_all=0.001,
        reg_all=0,
        init_mean=0,
        init_std_dev=0.01,
        verbose=True
    )


def create_train_logreg_model(df_boston_parsed_text: pd.DataFrame) -> LogisticRegression:
    """
    Creates the logistic regression model parsing first the required data
    """
    df_logreg = df_boston_parsed_text
    df_logreg['is_good'] = df_logreg.apply(lambda row: 1 if row.stars > 3 else 0, axis=1)
    x = df_logreg[['good_inter', 'good_diff', 'no_good_inter', 'no_good_diff']].to_numpy()
    y = df_logreg[['is_good']].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    log_regression = LogisticRegression(random_state=0).fit(x_train, y_train)
    return log_regression


def train_svd_model(model: spr.SVD, train: spr.Trainset):
    """
    Trains the SVD model
    """
    model.fit(trainset=train)
    return True


def predict_log_regression(user: str, log_regression: LogisticRegression):
    user_bus_df = pd.DataFrame(list(zip(itertools.repeat(user), list(dict_words_business.keys()))),
                               columns=['user_id', 'business_id'])
    user_bus_df['words_user_g'] = user_bus_df.apply(lambda row: dict_words_user[row.user_id]['0'], axis=1)
    user_bus_df['words_user_ng'] = user_bus_df.apply(lambda row: dict_words_user[row.user_id]['1'], axis=1)
    user_bus_df['words_business_g'] = user_bus_df.apply(lambda row: dict_words_business[row.business_id]['0'], axis=1)
    user_bus_df['words_business_ng'] = user_bus_df.apply(lambda row: dict_words_business[row.business_id]['1'], axis=1)
    user_bus_df['good_inter'] = user_bus_df.apply(lambda row: len(
        set(row['words_user_g'].strip('][').replace("'", '').split(', ')).intersection(
            set(row['words_business_g'].strip('][').replace("'", '').split(', ')))), axis=1)
    user_bus_df['good_diff'] = user_bus_df.apply(lambda row: len(
        set(row['words_user_g'].strip('][').replace("'", '').split(', ')).difference(
            set(row['words_business_g'].strip('][').replace("'", '').split(', ')))), axis=1)
    user_bus_df['no_good_inter'] = user_bus_df.apply(lambda row: len(
        set(row['words_user_ng'].strip('][').replace("'", '').split(', ')).intersection(
            set(row['words_business_ng'].strip('][').replace("'", '').split(', ')))), axis=1)
    user_bus_df['no_good_diff'] = user_bus_df.apply(lambda row: len(
        set(row['words_user_ng'].strip('][').replace("'", '').split(', ')).difference(
            set(row['words_business_ng'].strip('][').replace("'", '').split(', ')))), axis=1)
    X = user_bus_df[['good_inter', 'good_diff', 'no_good_inter', 'no_good_diff']].to_numpy()
    user_bus_df['result'] = pd.DataFrame(pd.Series(log_regression.predict(X).T))[0]
    return user_bus_df[['business_id', 'result']].set_index('business_id').to_dict()['result']


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((uid, iid, true_r, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[3], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def final_rating(user: str, svd_result: list, business_type: str):
    prediction = predict_log_regression(user,)
    return list(map(lambda x: x[3] * (0.8) if prediction.get(x[1]) == 0 else x[3], svd_result))

"""
===================
Evaluation metrics
===================
"""


def ndcg(predictions, verbose=True):
    """
    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
            This list of predictions only shows recomendations for a single user.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The normalized discounted cumulative gain of a list.
    Raises:
        ValueError: When ``predictions`` is empty.
    """

    # Let's make two copies of the prediction list in order to avoid reference errors while
    # modifying
    est_list = list(predictions)
    original_list = list(predictions)

    # Let's sort them
    est_list.sort(key=lambda tup: tup[3])
    original_list.sort(key=lambda tup: tup[2])

    # Select the common items
    # Disable due to we just want to verify an order, not the existence of an item
    index = []
    for i in range(len(original_list)):
        if original_list[i][1] == est_list[i][1]:
            index.append(i)
        elif len(index) != 0:
            break

    # Calculate the metric
    dcg = np.sum([(est_list[i][3] / log(2 + i)) for i in range(len(est_list))])
    idcg = 1 + np.sum([(original_list[i][2] / log(2 + i)) for i in range(len(original_list))])
    ndcg = dcg / idcg

    if verbose:
        print(f"nDCG: {ndcg}")

    return ndcg


def liftindex(predictions, verbose=True):
    """
        Args:
            predictions (:obj:`list` of :obj:`Prediction\
                <surprise.prediction_algorithms.predictions.Prediction>`):
                A list of predictions, as returned by the :meth:`test()
                <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
                This list of predictions only shows recomendations for a single user.
            verbose: If True, will print computed value. Default is ``True``.
        Returns:
            The liftindex of a list using percentiles. For this reason,
            the list must contain 10 elements.
        Raises:
            ValueError: When ``predictions`` is empty.
    """
    if len(predictions) == 0:
        return 0

    # Let's make two copies of the prediction list in order to avoid reference errors while
    # modifying
    est_list = list(predictions)
    original_list = list(predictions)

    # Let's sort them
    est_list.sort(key=lambda tup: tup[3])
    original_list.sort(key=lambda tup: tup[2])

    # Select the common items
    # Disable due to we just want to verify an order, not the existence of an item
    index = []
    for i in range(len(original_list)):
        if original_list[i][1] == est_list[i][1]:
            index.append(i)

    # No common items in place
    if len(index) == 0:
        return 0

    # Calculate the lift score using the match
    weight = 1 / len(est_list)  # Is the same as original_list, we just want the size of prediction
    liftindex = 0
    for idx in index:
        liftindex += est_list[idx][3] * (1 - weight * idx)

    if verbose:
        print(f"Lift index weight division: {weight}")
        print(f"Lift index matches: {index}")
        print(f"Lift index: {liftindex}")

    return liftindex


def single_list_similarity(predicted: list, feature_df: pd.DataFrame, u: int = -1) -> float:
    """
    Computes the intra-list similarity for a single list of recommendations.
    Parameters
    ----------
    predicted : a list
        Ordered predictions
        Example: ['X', 'Y', 'Z']
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
    ils_single_user: float
        The intra-list similarity for a single list of recommendations.
    """
    # exception predicted list empty
    if not (predicted):
        raise Exception('Predicted list is empty, index: {0}'.format(u))

    # get features for all recommended items
    recs_content = feature_df.loc[predicted]
    recs_content = recs_content.dropna()
    recs_content = sp.csr_matrix(recs_content.values)

    # calculate similarity scores for all items in list
    similarity = cosine_similarity(X=recs_content, dense_output=False)

    # get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    # calculate average similarity score of all recommended items in list
    ils_single_user = np.mean(similarity[upper_right])
    return ils_single_user
