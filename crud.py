"""
    Here, we are going to implement the logic to handle operations on the DB
    This CRUD function will be use on the view (endpoints) implementation
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from db import models
import schemas
import hashlib

# Hash plain text password
def hash_plain_password(password: str) -> str:
    return hashlib.sha1(bytes(password)).hexdigest()


# Get user's data only if the password provided is correct
def get_user(db: Session, user: schemas.UserAuth):
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
        if query_user.password is not hash_plain_password(user.password):
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
