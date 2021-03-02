"""
    Main module to set all views(endpoints) and launch all the backend.
    In this module, we create a connection to the DB and implement all routes
"""
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from db.database import SessionLocal, engine
from db import models
from sh.alembic_start import alembic_migration
import crud, schemas
import os
import uvicorn

# Create all tables in the DB when a migration is made
# This is currently made by Alembic, so dont worry
# models.Base.metadata.create_all(bind=engine)

# Instanciate the backend
app = FastAPI()

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


# Programatically start the server
if __name__ == "__main__":
    host, port = get_url()
    alembic_migration()
    uvicorn.run(app, host=host, port=port)