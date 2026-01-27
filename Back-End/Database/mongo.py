import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
import gridfs


load_dotenv()


def _required_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value.strip()


MONGO_URI = _required_env("MONGODB_URI")
DB_NAME = _required_env("MONGODB_DB")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """Create a singleton Mongo client."""
    return MongoClient(MONGO_URI)


def get_db():
    """Return the configured database."""
    return get_client()[DB_NAME]


@lru_cache(maxsize=1)
def get_fs() -> gridfs.GridFS:
    """GridFS instance for storing raw files."""
    return gridfs.GridFS(get_db())


def get_users_collection():
    return get_db()["users"]


def get_uploads_collection():
    return get_db()["uploads"]


def ensure_indexes() -> None:
    """Create simple indexes used by the API."""
    users = get_users_collection()
    users.create_index("email", unique=True)

    uploads = get_uploads_collection()
    uploads.create_index([("user_id", 1)])

