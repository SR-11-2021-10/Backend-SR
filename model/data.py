"""
Load data available in .csv files
"""
import pandas as pd
import surprise as spr
import numpy as np

DF_BOSTON = "data/df_boston.csv"
BUSINESS_DF = "data/business_df.csv"
BUSINESS_POS = "data/df_boston_businesspos.csv"
USER_POS = "data/df_boston_userpos.csv"


def load_df_boston() -> pd.DataFrame:
    """
    Loads boston reviews
    """
    return pd.read_csv(DF_BOSTON)


def parse_df_boston(df_boston: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the correct columns to parse the dataframe to the reader
    class of Scikit-Surprise
    """
    proyection = df_boston[['user_id', 'business_id', 'stars']]
    proyection['stars'] = proyection['stars'].astype('int32')
    return proyection


def load_business_df() -> pd.DataFrame:
    """
    Loads business aggregate dataframe. This dataframe includes
    the mean of the starts received and the funny, useful, and cool
    reviews
    """
    return pd.read_csv(BUSINESS_DF)


def load_business_pos_df() -> pd.DataFrame:
    """
    Loads the most important words describing a business
    """
    return pd.read_csv(BUSINESS_POS)


def load_user_pos_df() -> pd.DataFrame:
    """
    Loads the most important words that an user use to write reviews
    """
    return pd.read_csv(USER_POS)


def load_data_surprise_format(proyection: pd.DataFrame) -> tuple:
    """
    Split the proyection dataframe with all the information into the train,
    validation and test groups.
    """
    reader = spr.Reader(rating_scale=(1, 5))
    train, validate, test = np.split(proyection.sample(frac=1), [int(.6 * len(proyection)), int(.8 * len(proyection))])

    train_data = spr.Dataset.load_from_df(train, reader)
    validation_data = spr.Dataset.load_from_df(validate, reader)
    test_data = spr.Dataset.load_from_df(test, reader)

    train_data = train_data.build_full_trainset()
    validation_data = validation_data.build_full_trainset()
    test_data = test_data.build_full_trainset()

    validation_data = validation_data.build_testset()
    test_data = test_data.build_testset()

    # Return the data
    return train_data, validation_data, test_data


def parse_text_model_df(df_boston: pd.DataFrame, df_words_user: pd.DataFrame,
                        df_words_business: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the data required by the text model.
    """
    df_boston['key'] = df_boston['user_id'] + df_boston['business_id']
    dict_words_user = df_words_user.set_index('Unnamed: 0').to_dict('index')
    dict_words_business = df_words_business.set_index('Unnamed: 0').to_dict('index')
    dict_stars = df_boston[['key', 'stars']].set_index('key').to_dict()['stars']

    df_boston_parsed = df_boston[['user_id', 'business_id', 'stars']]

    df_boston_parsed['words_user_g'] = df_boston_parsed.apply(lambda row: dict_words_user[row.user_id]['0'], axis=1)
    df_boston_parsed['words_user_ng'] = df_boston_parsed.apply(lambda row: dict_words_user[row.user_id]['1'], axis=1)
    df_boston_parsed['words_business_g'] = df_boston_parsed.apply(lambda row: dict_words_business[row.business_id]['0'],
                                                                  axis=1)
    df_boston_parsed['words_business_ng'] = df_boston_parsed.apply(
        lambda row: dict_words_business[row.business_id]['1'], axis=1)
    df_boston_parsed['good_inter'] = df_boston_parsed.apply(lambda row: len(
        set(row['words_user_g'].strip('][').replace("'", '').split(', ')).intersection(
            set(row['words_business_g'].strip('][').replace("'", '').split(', ')))), axis=1)
    df_boston_parsed['good_diff'] = df_boston_parsed.apply(lambda row: len(
        set(row['words_user_g'].strip('][').replace("'", '').split(', ')).difference(
            set(row['words_business_g'].strip('][').replace("'", '').split(', ')))), axis=1)
    df_boston_parsed['no_good_inter'] = df_boston_parsed.apply(lambda row: len(
        set(row['words_user_ng'].strip('][').replace("'", '').split(', ')).intersection(
            set(row['words_business_ng'].strip('][').replace("'", '').split(', ')))), axis=1)
    df_boston_parsed['no_good_diff'] = df_boston_parsed.apply(lambda row: len(
        set(row['words_user_ng'].strip('][').replace("'", '').split(', ')).difference(
            set(row['words_business_ng'].strip('][').replace("'", '').split(', ')))), axis=1)
    df_boston_parsed['stars'] = df_boston_parsed.apply(lambda row: dict_stars.get(row.user_id + row.business_id),
                                                       axis=1)

    return df_boston_parsed, dict_words_user, dict_words_business, dict_stars


