# Invokes two shell commands to migrate database scheme using Alembic
import subprocess


def alembic_migration():
    """
    Executes alembic migration
    """
    # Create revision
    subprocess.run(f'alembic%revision%--autogenerate%-m%"Docker revision"'.split("%"))
    # Upgrade migration
    subprocess.run(f"alembic%upgrade%head".split("%"))
