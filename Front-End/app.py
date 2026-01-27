import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AIDEX Upload Demo", page_icon="🗂️", layout="centered")
st.title("AIDEX demo: sign up, sign in, upload a dataset")
st.caption("Backend: %s" % BACKEND_URL)

if "user" not in st.session_state:
    st.session_state["user"] = None  
if "files" not in st.session_state:
    st.session_state["files"] = []  


def show_error(message: str) -> None:
    st.error(message)


def api_post(
    path: str,
    *,
    json: Dict[str, Any] | None = None,
    data=None,
    files=None,
    timeout: float = 30.0,
):
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}",
            json=json,
            data=data,
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:  
        show_error(f"Request failed: {exc}")
        return None


def fetch_files() -> None:
    user = st.session_state.get("user")
    if not user:
        return
    try:
        resp = requests.get(f"{BACKEND_URL}/files/{user['user_id']}", timeout=15)
        resp.raise_for_status()
        st.session_state["files"] = resp.json().get("files", [])
    except requests.RequestException as exc:  # noqa: BLE001
        show_error(f"Could not load uploads: {exc}")


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

