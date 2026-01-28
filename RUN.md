## Quick Run

1) Create `.env` in project root:
```
MONGODB_URI=your-mongo-uri
MONGODB_DB=aidex
BACKEND_URL=http://localhost:8000
```

2) Install deps:
```
python -m pip install -r requirements.txt 
```

3) Start backend (from project root):
```
python -m uvicorn main:app --reload
```

4) Start frontend (new shell, project root):
```
python -m streamlit run Front-End/app.py
```

Open the Streamlit URL (default http://localhost:8501), sign up/sign in, and upload a file. It saves directly to Mongo (GridFS).***

