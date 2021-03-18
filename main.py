"""
    Main module to set all views(endpoints) and launch all the backend.
    In this module, we create a connection to the DB and implement all routes
"""
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import parse_obj_as
from sqlalchemy.orm import Session
from db.database import SessionLocal, engine
from db import models
from sh.alembic_start import alembic_migration
import sr.load_data
import crud, schemas
import os
import uvicorn

# Create all tables in the DB when a migration is made
# This is currently made by Alembic, so dont worry
# models.Base.metadata.create_all(bind=engine)

# Instanciate the backend
app = FastAPI()

# Load and save pandas dataframe
ratings = None

# Load and save artist info
artist = None

# Keep the artist names in memory to speed up the retrieve
artist_db = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
# Create a new connection to handle data
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_url():
    """
    Get server's host and port
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", 8000)
    return host, port


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserAuth, db: Session = Depends(get_db)):
    """
    Endpoint to create a new user
    """
    return crud.create_user(db=db, user=user)


@app.post("/login/", response_model=schemas.User)
def login_user(user: schemas.UserAuth, db: Session = Depends(get_db)):
    """
    Enpoint to retrieve a specific user data
    """
    return crud.login_user(db=db, user=user)


@app.post("/recommend")
def make_recommendation(
    recommendation: schemas.Recommendation, db: Session = Depends(get_db)
):
    """
    Endpoint to retrieve a recommendation
    """
    return crud.make_recommendation(
        data=ratings, recommendation=recommendation, artist=artist, db=db
    )


@app.get("/artist")
def get_all_artists(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all artist information
    """
    global artist_db
    if artist_db == None:
        artist_model = crud.get_all_artist(db=db)
        pydantic_response = parse_obj_as(List[schemas.Artist], artist_model)
        response = [r.dict() for r in pydantic_response]
        artist_db = response
        return response
    else:
        return artist_db


@app.get("/ratings/{username}")
def get_user_ratings(username: str, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all user ratings
    """

    user_ratings = crud.get_user_ratings(db=db, username=username)
    response = [
        schemas.RatingResponse(
            user=rating_data.user,
            item=rating_data.item,
            rating=rating_data.rating,
            artist_name=artist_data.artist_name,
        )
        for rating_data, artist_data in user_ratings
    ]
    return response


@app.post("/ratings/")
def create_rating(ratings: List[schemas.Rating], db: Session = Depends(get_db)):
    """
    Endpoint to create a rating
    """
    return crud.create_rating(ratings=ratings, db=db)


# Programatically start the server
if __name__ == "__main__":
    # Load pandas dataframe (Ratings)
    ratings = sr.load_data.load_data(sr.load_data.RATINGS, sep=",")
    # Load artist dataframe (Artist)
    artist = sr.load_data.load_data(sr.load_data.ARTIST, sep="\t")
    host, port = get_url()
    alembic_migration()
    uvicorn.run(app, host=host, port=port)
