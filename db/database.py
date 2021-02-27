# DB Access config
""" 
    DB Access Config
    Here, SQLAlchemy is used as a ORM to easily access and manipulate the relational DB.
    If you want more details about how to connect to databases using FastAPI see:
    https://fastapi.tiangolo.com/tutorial/sql-databases/?h=sql#orms
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB URI Connection
# In this moment, I will work with my postgresql local db. Later, we will change it
# to work with postgres container db
SQLALCHEMY_DATABASE_URL = "postgresql://ggonzr:ggonzr@localhost:5432/ggonzr"

# Define the DB Engine the ORM will work with
engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)

# Define the parameters for each DB session. This is only the definition
# not the session. To create the session instanciate SessionLocal()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base ORM Model
Base = declarative_base()