# DB models used by the ORM
"""
    DB models used by the ORM
    Currently, we are not going to use relations in the DB
    SQLAlchemy handles this in a simple and easy way. Please read the next reference
    https://fastapi.tiangolo.com/tutorial/sql-databases/?h=sql#create-the-database-models
"""
from sqlalchemy import Boolean, Column, Integer, String
from .database import Base

# Base is the base model we instanciate in database.py module
# Now we are going to create the database models


class User(Base):
    """
    User model. This represent the basic user info only with
    his/her username and password
    """

    # Table name in DB
    __tablename__ = "users"

    # Table Fields
    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)