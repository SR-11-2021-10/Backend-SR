"""
    Here, we are going to instanciate the schemas to help us
    validate HTTP body request has an appropiate format.    
"""

# Pydantic Base Schema
from pydantic import BaseModel, Field


class UserBase(BaseModel):
    """
    This class only includes the username attributes which is the common attributes
    when we receive a POST to create a new one or GET an existing one.
    """

    username: str


class UserAuth(UserBase):
    """
    This schema allows us to parse the content off all client request including the password
    """

    password: str


class User(UserBase):
    """
    This is the schema we will use to response a request about an user information
    Config subclass indicates pydantic to retrieve data from ORM models. Note that
    we are not retriving the hashed_password field in the model to send in response.
    This is the main reason we create 3 schemas
    """

    # Now the User PK is the username
    # For this reason, a id PK is not longer needed
    # id: int

    class Config:
        orm_mode = True


class Recommendation(BaseModel):
    """
    Request body to make a recommendation

    Attributes
    ----------
    type : str
        Type of model to create and predict
    similitude : str
        Similitude measure to use
    username : str
        Username to make the recommendation
    artist : str
        Artist id to predict
    """

    type: str
    similitude: str
    username: str
    artist: str


class Artist(BaseModel):
    """
    Artist Response Schema
    """

    id: str = Field(..., alias="artist_id")
    name: str = Field(..., alias="artist_name")

    class Config:
        orm_mode = True
