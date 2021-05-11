"""
    Here, we are going to instanciate the schemas to help us
    validate HTTP body request has an appropiate format.    
"""

# Pydantic Base Schema
from pydantic import BaseModel, Field
from typing import Optional


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
    business : str
        Type of business to filter
    starts: int
        Minimum numbers of starts to filter the recomendation
    username : str
        Username to make the recommendation
    """

    business: str
    starts: int
    username: str


class Artist(BaseModel):
    """
    Artist Response Schema
    """

    id: str = Field(alias="artist_id")
    name: str = Field(alias="artist_name")
    # artist_id: Optional[str]
    # artist_name: Optional[str]

    class Config:
        orm_mode = True


class Rating(BaseModel):
    user: str
    item: str
    rating: int


class RatingResponse(Rating):
    artist_name: str


class RatingModel(Rating):
    class Config:
        orm_mode = True