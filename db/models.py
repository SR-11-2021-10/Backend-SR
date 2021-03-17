# DB models used by the ORM
"""
    DB models used by the ORM
    Currently, we are not going to use relations in the DB
    SQLAlchemy handles this in a simple and easy way. Please read the next reference
    https://fastapi.tiangolo.com/tutorial/sql-databases/?h=sql#create-the-database-models
"""
from sqlalchemy import Boolean, Column, Integer, String, Float
from sqlalchemy.orm import relationship
from .database import Base

# Base is the base model we instanciate in database.py module
# Now we are going to create the database models


class User(Base):
    """
    User model. This represent the basic user info only with
    his/her username and password
    """

    # Table name in DB
    __tablename__ = "user"

    # Table Fields
    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)
    gender = Column(String(2))
    age = Column(Integer)
    country = Column(String(250))


#    username = Column(String, primary_key=True, index=True)
#    hashed_password = Column(String)


class Artist(Base):
    """
    Artist model. This represent the basic artist info
    """

    # Table name in DB
    __tablename__ = "artist"

    # Table Fields
    artist_id = Column(String(250), primary_key=True, index=True)
    artist_name = Column(String(250), nullable=False)


class Rating(Base):
    """
    Artist model. This represent the basic artist info
    """

    # Table name in DB
    __tablename__ = "rating"

    # Table Fields
    user_id = relationship("User", back_populates="username")
    artist_id = relationship("Artist", back_populates="artist_id")
    rating = Column(Integer, nullable=False)


class Estimation(Base):
    """
    Artist model. This represent the basic artist info
    """

    # Table name in DB
    __tablename__ = "estimation"

    # Table Fields
    user_id = relationship("User", back_populates="username")
    artist_id = relationship("Artist", back_populates="artist_id")
    estimation = Column(Float, nullable=False)