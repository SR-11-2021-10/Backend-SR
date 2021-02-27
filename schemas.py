"""
    Here, we are going to instanciate the schemas to help us
    validate HTTP body request has an appropiate format.    
"""

# Pydantic Base Schema
from pydantic import BaseModel


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

    id: int

    class Config:
        orm_mode = True