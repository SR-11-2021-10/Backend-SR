# Import all modules together to be catched by alembic
# to make full migration
from .database import Base
from . import models