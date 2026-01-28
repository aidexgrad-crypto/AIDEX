import datetime
import getpass
import hashlib
from typing import Any, Dict, List

from bson import ObjectId
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gridfs.errors import NoFile
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
    role: str = "user"


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
def require_admin_user(user_id: str) -> Dict[str, Any]:
    user = require_user(user_id)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def admin_exists() -> bool:
    return bool(get_users_collection().find_one({"role": "admin"}))


def create_admin(email: str, password: str) -> None:
    email = normalize_email(email)
    get_users_collection().create_index("email", unique=True)
    try:
        get_users_collection().insert_one(
            {
                "email": email,
                "password_hash": hash_password(password),
                "created_at": datetime.datetime.utcnow(),
                "role": "admin",
            }
        )
        print(f"[admin created] {email}")
    except DuplicateKeyError:
        print(f"[admin exists] {email}")


def ensure_admin_interactive() -> None:
    if admin_exists():
        return
    print("\nNo admin user found. Create one now.")
    email = input("Admin email: ").strip()
    if not email:
        raise RuntimeError("Admin email is required.")
    password = getpass.getpass("Admin password (input hidden): ").strip()
    if not password:
        raise RuntimeError("Admin password is required.")
    create_admin(email, password)


def iso(dt: datetime.datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None


@app.on_event("startup")
def _startup() -> None:
    ensure_indexes()
    ensure_admin_interactive()


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
                "role": "user",
            }
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")

    return UserResponse(user_id=str(result.inserted_id), email=email, role="user")


@app.post("/signin", response_model=UserResponse)
def signin(payload: SignInRequest) -> UserResponse:
    email = normalize_email(payload.email)
    user = get_users_collection().find_one({"email": email})
    if not user or user["password_hash"] != hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return UserResponse(user_id=str(user["_id"]), email=email, role=user.get("role", "user"))


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


# -------- Admin endpoints --------
@app.get("/admin/users")
def admin_list_users(x_admin_user: str | None = Header(None)) -> Dict[str, List[Dict[str, Any]]]:
    require_admin_user(x_admin_user)
    uploads = get_uploads_collection()
    out: List[Dict[str, Any]] = []
    for user in get_users_collection().find().sort("created_at", -1):
        user_id = str(user["_id"])
        out.append(
            {
                "user_id": user_id,
                "email": user["email"],
                "created_at": iso(user.get("created_at")),
                "uploads": uploads.count_documents({"user_id": user_id}),
                "role": user.get("role", "user"),
            }
        )
    return {"users": out}


@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: str, x_admin_user: str | None = Header(None)) -> Dict[str, Any]:
    require_admin_user(x_admin_user)
    user = require_user(user_id)

    uploads = list(get_uploads_collection().find({"user_id": user_id}, {"_id": 1}))
    for doc in uploads:
        try:
            get_fs().delete(doc["_id"])
        except NoFile:
            pass
    get_uploads_collection().delete_many({"user_id": user_id})
    get_users_collection().delete_one({"_id": user["_id"]})
    return {"deleted": True, "removed_files": len(uploads)}


@app.get("/admin/uploads")
def admin_list_uploads(
    user_id: str | None = None,
    x_admin_user: str | None = Header(None),
) -> Dict[str, List[Dict[str, Any]]]:
    require_admin_user(x_admin_user)
    query: Dict[str, Any] = {}
    if user_id:
        query["user_id"] = user_id
    uploads = []
    for doc in get_uploads_collection().find(query).sort("uploaded_at", -1):
        uploads.append(
            {
                "file_id": str(doc["_id"]),
                "user_id": doc["user_id"],
                "filename": doc["filename"],
                "uploaded_at": iso(doc.get("uploaded_at")),
            }
        )
    return {"uploads": uploads}


@app.delete("/admin/uploads/{file_id}")
def admin_delete_upload(file_id: str, x_admin_user: str | None = Header(None)) -> Dict[str, Any]:
    require_admin_user(x_admin_user)
    try:
        object_id = ObjectId(file_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid file id") from exc

    doc = get_uploads_collection().find_one({"_id": object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Upload not found")

    get_uploads_collection().delete_one({"_id": object_id})
    try:
        get_fs().delete(object_id)
    except NoFile:
        pass

    return {"deleted": True, "file_id": file_id}

