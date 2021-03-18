"""
    Here, we are going to implement the logic to handle operations on the DB
    This CRUD function will be use on the view (endpoints) implementation
"""
from fastapi import HTTPException
from typing import List
from pydantic import parse_obj_as
from sqlalchemy.orm import Session
from db import models
from sr.model import RecommenderSystem
import pandas as pd
import schemas
import hashlib

# Hash plain text password
def hash_plain_password(password: str) -> str:
    return hashlib.sha1(bytes(password, "UTF-8")).hexdigest()


# Get user's data only if the password provided is correct
def login_user(db: Session, user: schemas.UserAuth):
    # Retrieve the user
    query_user = (
        db.query(models.User).filter(models.User.username == user.username).first()
    )
    if query_user is None:
        # User does not exist
        raise HTTPException(
            status_code=404, detail=f"User: {user.username} does not exist !"
        )
    else:
        hashed_pass = hash_plain_password(user.password)
        user_pass = query_user.hashed_password
        if user_pass != hashed_pass:
            raise HTTPException(status_code=403, detail="Wrong password")
        else:
            return query_user


def create_user(db: Session, user: schemas.UserAuth):
    hash_password = hash_plain_password(user.password)
    # New instance of User
    db_user = models.User(username=user.username, hashed_password=hash_password)
    # Create new user
    db.add(db_user)
    # Commit transaction
    db.commit()
    # Get created user with ID to send into response
    db.refresh(db_user)
    return db_user


def make_recommendation(
    data: pd.DataFrame,
    recommendation: schemas.Recommendation,
    artist: pd.DataFrame,
    db: Session,
):
    # Retrieves current ratings data
    current_ratings = retrieve_all_ratings(db=db)

    # Create the recommendation model
    rs = RecommenderSystem(
        type=recommendation.type,
        similitude=recommendation.similitude,
        user_item_rating=current_ratings,
    )
    try:
        prediction = rs.predict(
            uid=recommendation.username, iid=recommendation.artist, artist=artist
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail={"msg": f"{e}"})


def get_all_artist(db: Session):
    try:
        artist = db.query(models.Artist).all()
        return artist
    except Exception as e:
        raise HTTPException(status_code=500, detail={"msg": f"{e}"})


def get_user_ratings(db: Session, username: str):
    try:
        ratings = (
            db.query(models.Rating, models.Artist)
            .filter(
                models.Rating.item == models.Artist.artist_id,
                models.Rating.user == username,
            )
            .all()
        )
        print("Crud response: ", ratings)
        return ratings
    except Exception as e:
        print("[Crud][get_user_ratings] Error: ", e)
        raise HTTPException(status_code=500, detail={"msg": f"{e}"})


def create_rating(ratings: List[schemas.Rating], db: Session):
    for rating in ratings:
        db_ratings = models.Rating(**rating.dict())
        db.add(db_ratings)
    # Commit transaction
    db.commit()
    return {"msg": "Operation complete"}


def retrieve_all_ratings(db: Session) -> pd.DataFrame:
    """
    Retrieves all available in DB to be used to train the model

    Returns
    --------
    pd.DataFrame:
        Pandas dataframe with all ratings available to train the model
    """
    try:
        ratings_orm = db.query(models.Rating).all()
        ratings_pydantic = parse_obj_as(List[schemas.RatingModel], ratings_orm)
        ratings_to_pandas = []
        for r in ratings_pydantic:
            r_dict = r.dict()
            rating_to_list = [r_dict["user"], r_dict["item"], r_dict["rating"]]
            ratings_to_pandas.append(rating_to_list)
        print("[crud][retrieve_all_ratings] Current ratings retrieved")
        return pd.DataFrame(data=ratings_to_pandas, columns=["user", "item", "rating"])
    except Exception as e:
        print("[Crud][retrieve_all_ratings] Error: ", e)
        raise HTTPException(status_code=500, detail={"msg": f"{e}"})