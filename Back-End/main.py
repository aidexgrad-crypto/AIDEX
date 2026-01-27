import datetime
import hashlib
from typing import Any, Dict, List

from bson import ObjectId
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from pymongo.errors import DuplicateKeyError

from Database.mongo import ensure_indexes, get_fs, get_uploads_collection, get_users_collection


app = FastAPI(title="AIDEX Auth + Upload Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    user_id: str
    email: EmailStr


def hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def require_user(user_id: str) -> Dict[str, Any]:
    try:
        object_id = ObjectId(user_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid user id") from exc

    user = get_users_collection().find_one({"_id": object_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def iso(dt: datetime.datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None


@app.on_event("startup")
def _startup() -> None:
    ensure_indexes()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/signup", response_model=UserResponse)
def signup(payload: SignUpRequest) -> UserResponse:
    email = normalize_email(payload.email)
    try:
        result = get_users_collection().insert_one(
            {
                "email": email,
                "password_hash": hash_password(payload.password),
                "created_at": datetime.datetime.utcnow(),
            }
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")

    return UserResponse(user_id=str(result.inserted_id), email=email)


@app.post("/signin", response_model=UserResponse)
def signin(payload: SignInRequest) -> UserResponse:
    email = normalize_email(payload.email)
    user = get_users_collection().find_one({"email": email})
    if not user or user["password_hash"] != hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return UserResponse(user_id=str(user["_id"]), email=email)


@app.post("/upload")
async def upload_dataset(
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    _ = require_user(user_id)

    file_bytes = await file.read()
    uploaded_at = datetime.datetime.utcnow()

    file_id = get_fs().put(
        file_bytes,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        metadata={"user_id": user_id, "uploaded_at": uploaded_at},
    )

    get_uploads_collection().insert_one(
        {
            "_id": file_id,
            "user_id": user_id,
            "filename": file.filename,
            "content_type": file.content_type or "application/octet-stream",
            "uploaded_at": uploaded_at,
        }
    )

    return {
        "file_id": str(file_id),
        "user_id": user_id,
        "filename": file.filename,
        "uploaded_at": iso(uploaded_at),
    }


@app.get("/files/{user_id}")
def list_files(user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    _ = require_user(user_id)

    uploads = get_uploads_collection()
    files = []
    for doc in uploads.find({"user_id": user_id}).sort("uploaded_at", -1):
        files.append(
            {
                "file_id": str(doc["_id"]),
                "filename": doc["filename"],
                "uploaded_at": iso(doc.get("uploaded_at")),
            }
        )
    return {"files": files}

