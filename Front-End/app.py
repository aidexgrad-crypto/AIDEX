import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AIDEX Upload/Admin Demo", page_icon="🗂️", layout="wide")
st.title("AIDEX demo")
st.caption(f"Backend: {BACKEND_URL}")

if "user" not in st.session_state:
    st.session_state["user"] = None
if "files" not in st.session_state:
    st.session_state["files"] = []
if "admin" not in st.session_state:
    st.session_state["admin"] = {"user": None, "users": [], "uploads": [], "selected_user": None}


def show_error(message: str) -> None:
    st.error(message)


def api_request(
    method: str,
    path: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Dict[str, Any] | None = None,
    data=None,
    files=None,
    timeout: float = 30.0,
):
    try:
        resp = requests.request(
            method,
            f"{BACKEND_URL}{path}",
            headers=headers,
            params=params,
            json=json,
            data=data,
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.content
    except requests.RequestException as exc:  # noqa: BLE001
        show_error(f"{method.upper()} {path} failed: {exc}")
        return None


def api_post(path: str, **kwargs):
    return api_request("post", path, **kwargs)


def api_get(path: str, **kwargs):
    return api_request("get", path, **kwargs)


def api_delete(path: str, **kwargs):
    return api_request("delete", path, **kwargs)


def fetch_files() -> None:
    user = st.session_state.get("user")
    if not user:
        return
    resp = api_get(f"/files/{user['user_id']}")
    if resp:
        st.session_state["files"] = resp.get("files", [])


def load_admin_users():
    admin = st.session_state["admin"]
    if not admin["user"]:
        show_error("Admin not signed in.")
        return
    res = api_get("/admin/users", headers={"X-Admin-User": admin["user"]["user_id"]})
    if res:
        admin["users"] = res.get("users", [])


def load_admin_uploads(user_filter: str | None = None):
    admin = st.session_state["admin"]
    if not admin["user"]:
        show_error("Admin not signed in.")
        return
    params = {"user_id": user_filter} if user_filter else None
    res = api_get("/admin/uploads", headers={"X-Admin-User": admin["user"]["user_id"]}, params=params)
    if res:
        admin["uploads"] = res.get("uploads", [])
        admin["selected_user"] = user_filter


mode = st.sidebar.radio("Mode", ["User", "Admin"], index=0)

if mode == "User":
    signup_col, signin_col = st.columns(2)

    with signup_col:
        st.subheader("Create account")
        with st.form("signup_form"):
            su_email = st.text_input("Email", key="signup_email")
            su_password = st.text_input("Password", type="password", key="signup_password")
            su_submit = st.form_submit_button("Sign up")
            if su_submit:
                if not su_email or not su_password:
                    show_error("Email and password are required.")
                else:
                    res = api_post("/signup", json={"email": su_email, "password": su_password})
                    if res:
                        st.success("Account created. You can sign in now.")

    with signin_col:
        st.subheader("Sign in")
        with st.form("signin_form"):
            si_email = st.text_input("Email", key="signin_email")
            si_password = st.text_input("Password", type="password", key="signin_password")
            si_submit = st.form_submit_button("Sign in")
            if si_submit:
                res = api_post("/signin", json={"email": si_email, "password": si_password})
                if res:
                    st.session_state["user"] = res
                    fetch_files()
                    st.success(f"Signed in as {res['email']}")

    user = st.session_state.get("user")
    if user:
        st.divider()
        cols = st.columns([3, 1])
        cols[0].success(f"Signed in as {user['email']}")
        if cols[1].button("Log out"):
            st.session_state["user"] = None
            st.session_state["files"] = []
            st.experimental_rerun()

        st.subheader("Upload a dataset")
        dataset = st.file_uploader(
            "Pick a file from your laptop",
            type=["csv", "xlsx", "json", "txt", "zip", "parquet", "pkl"],
            accept_multiple_files=False,
        )
        if dataset and st.button("Upload"):
            files = {"file": (dataset.name, dataset.getvalue(), dataset.type or "application/octet-stream")}
            data = {"user_id": user["user_id"]}
            res = api_post("/upload", data=data, files=files, timeout=300)
            if res:
                st.success(f"Uploaded {res['filename']} (id: {res['file_id']})")
                fetch_files()

        st.subheader("Previous uploads")
        if not st.session_state["files"]:
            st.info("No uploads yet.")
        else:
            for file_doc in st.session_state["files"]:
                st.write(f"- {file_doc['filename']} (at {file_doc.get('uploaded_at', 'unknown')})")
    else:
        st.info("Sign in to upload and view your datasets.")


# ---------------- Admin mode ----------------
else:
    admin = st.session_state["admin"]
    st.subheader("Admin panel")

    with st.form("admin_signin_form"):
        ad_email = st.text_input("Admin email")
        ad_password = st.text_input("Admin password", type="password")
        ad_submit = st.form_submit_button("Sign in as admin")
        if ad_submit:
            res = api_post("/signin", json={"email": ad_email, "password": ad_password})
            if res and res.get("role") == "admin":
                admin["user"] = res
                st.success(f"Signed in as admin {res['email']}")
            else:
                show_error("Invalid admin credentials or not an admin account.")

    if not admin["user"]:
        st.info("Sign in with an admin account to manage users and uploads.")
    else:
        cols = st.columns(2)
        if cols[0].button("Load users"):
            load_admin_users()
        if cols[1].button("Load all uploads"):
            load_admin_uploads()

        st.markdown("### Users")
        if not admin["users"]:
            st.info("No users loaded yet.")
        else:
            for user in admin["users"]:
                row = st.columns([5, 1, 1])
                row[0].write(f"{user['email']} • uploads: {user['uploads']} • created: {user.get('created_at', 'n/a')} • role: {user.get('role', 'user')}")
                if row[1].button("Uploads", key=f"view_{user['user_id']}"):
                    load_admin_uploads(user["user_id"])
                if row[2].button("Delete", key=f"del_{user['user_id']}"):
                    res = api_delete(
                        f"/admin/users/{user['user_id']}",
                        headers={"X-Admin-User": admin["user"]["user_id"]},
                    )
                    if res:
                        st.success(f"Deleted {user['email']}")
                        load_admin_users()

        st.markdown("### Uploads")
        if admin["selected_user"]:
            st.caption(f"Filtered by user: {admin['selected_user']}")
        if not admin["uploads"]:
            st.info("No uploads loaded yet.")
        else:
            for upload in admin["uploads"]:
                row = st.columns([5, 1])
                row[0].write(f"{upload['filename']} • user: {upload['user_id']} • at {upload.get('uploaded_at', 'n/a')}")
                if row[1].button("Delete file", key=f"del_file_{upload['file_id']}"):
                    res = api_delete(
                        f"/admin/uploads/{upload['file_id']}",
                        headers={"X-Admin-User": admin["user"]["user_id"]},
                    )
                    if res:
                        st.success(f"Deleted {upload['filename']}")
                        load_admin_uploads(admin["selected_user"])

