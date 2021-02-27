"""
    Main module to set all views(endpoints) and launch all the backend.
    In this module, we create a connection to the DB and implement all routes
"""
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import crud, models, schemas
import uvicorn

# Create all tables in the DB when a migration is made
models.Base.metadata.create_all(bind=engine)

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


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserBase, db: Session = Depends(get_db)):
    """
    Endpoint to create a new user
    """
    return crud.create_user(db=db, user=user)


@app.get("/login/", response_model=schemas.User)
def get_user(user: schemas.UserBase, db: Session = Depends(get_db)):
    """
    Enpoint to retrieve a specific user data
    """
    return crud.get_user(db=db, user=user)


# Programatically start the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)