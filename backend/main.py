# main.py
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from upload import upload_csv
from query import ask_question
from csv_analysis import analyze_csv, generate_graphs
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, select, create_engine as sqlmodel_create_engine

app = FastAPI(title="CSV Analysis & Chatbot System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for graphs)
os.makedirs("graphs", exist_ok=True)
app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")

# Include routers
app.include_router(upload_csv, prefix="/api")
app.include_router(ask_question, prefix="/api")

# In-memory storage for chat histories per table
chat_histories: Dict[str, List[Dict]] = {}


# ENSURE chat history is temporary and only for the active table:
chat_history = []
active_table = None

def set_active_table(table_name):
    global active_table, chat_history
    active_table = table_name
    chat_history = []  # Clear chat history



@app.post("/api/generate_report/")
async def generate_report(table_name: str = None):
    """
    Generate a comprehensive report for the active table or specified table
    """
    try:
        from upload import get_current_session
        session = get_current_session()
        
        # Determine which table to analyze
        target_table = table_name if table_name else session.get("active_table")
        
        if not target_table:
            raise HTTPException(
                status_code=400, 
                detail="No active table found. Please upload a CSV file first."
            )
        
        # Get table info
        table_info = session.get("table_info", {})
        
        # Create a temporary CSV file for analysis
        # This is needed because analyze_csv expects a file path
        from sqlalchemy import create_engine, text
        from dotenv import load_dotenv
        
        load_dotenv()
        MYSQL_URI = os.getenv("MYSQL_URI")
        engine = create_engine(MYSQL_URI)
        
        # Export table data to DataFrame
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {target_table}", conn)
        
        # Create temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{target_table}_temp.csv")
        df.to_csv(temp_file, index=False)
        
        # Generate analysis
        summary, outliers_json, graph_paths = analyze_csv(temp_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Prepare response with relative paths for frontend
        graph_urls = [f"/graphs/{os.path.basename(path)}" for path in graph_paths]
        
        return {
            "table_name": target_table,
            "original_name": table_info.get("original_name", target_table),
            "file_name": table_info.get("file_name", "N/A"),
            "summary": summary,
            "outliers": json.loads(outliers_json) if outliers_json else {},
            "graph_urls": graph_urls,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/api/switch_table/")
def switch_active_table(table_name: str):
    """
    Switch the active table and reset chat history/UI.
    """
    global active_table, chat_history
    try:
        from upload import get_current_session
        from sqlalchemy import create_engine, text
        from dotenv import load_dotenv
        
        load_dotenv()
        MYSQL_URI = os.getenv("MYSQL_URI")
        engine = create_engine(MYSQL_URI)
        
        # Verify table exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'"))
            if not result.fetchall():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
            
            # Get table info
            columns_result = conn.execute(text(f"DESCRIBE {table_name}"))
            columns = [row[0] for row in columns_result.fetchall()]
            
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = count_result.fetchone()[0]
        
        # Update session
        session = get_current_session()
        session["active_table"] = table_name
        session["table_info"] = {
            "original_name": table_name,
            "safe_name": table_name,
            "columns": columns,
            "rows": row_count,
            "switched_at": datetime.now().isoformat()
        }
        
        # Reset chat history when switching table
        chat_history = []

        return {
            "message": f"Switched to table '{table_name}'",
            "active_table": table_name,
            "table_info": session["table_info"],
            "chat_history": []  # Always empty after switch
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching table: {str(e)}")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Fetch user from DB
    load_dotenv()
    MYSQL_URI = os.getenv("MYSQL_URI")
    engine = create_engine(MYSQL_URI)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT username FROM users WHERE username=:username"), {"username": username})
        user = result.fetchone()
        if not user:
            raise credentials_exception
    return username

# Sidebar endpoint: just show table names from MySQL
@app.get("/api/list_tables/")
async def list_tables(current_user: str = Depends(get_current_user)):
    """
    List available tables for sidebar.
    """
    try:
        from upload import get_current_session
        session = get_current_session()
        
        table_names, _ = get_schema_info()
        return {
            "tables": table_names,
            "active_table": session["active_table"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

 
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta=None):
    from datetime import datetime, timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# 1. Define the User model
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str

# 2. Create the engine and tables (do this once at startup)
load_dotenv()
MYSQL_URI = os.getenv("MYSQL_URI")
engine = sqlmodel_create_engine(MYSQL_URI, echo=True)

SQLModel.metadata.create_all(engine)

# 3. Update the signup endpoint
class SignupRequest(BaseModel):
    username: str
    password: str

@app.post("/api/signup/")
async def signup(payload: SignupRequest):
    """
    Register a new user. Stores username and hashed password in the database.
    """
    username = payload.username
    password = payload.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    hashed_password = get_password_hash(password)

    with Session(engine) as session:
        # Check if user already exists
        statement = select(User).where(User.username == username)
        user = session.exec(statement).first()
        if user:
            raise HTTPException(status_code=400, detail="Username already registered")
        # Insert new user
        new_user = User(username=username, hashed_password=hashed_password)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
    return {"message": "User registered successfully"}

@app.post("/api/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    User login. Returns JWT token if credentials are correct.
    """
    load_dotenv()
    MYSQL_URI = os.getenv("MYSQL_URI")
    engine = create_engine(MYSQL_URI)
    username = form_data.username
    password = form_data.password

    with engine.connect() as conn:
        result = conn.execute(text("SELECT username, hashed_password FROM users WHERE username=:username"), {"username": username})
        user = result.fetchone()
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/my_tables/")
async def get_my_tables(current_user: str = Depends(get_current_user)):
    """
    Return only the tables for the authenticated user.
    """
    load_dotenv()
    MYSQL_URI = os.getenv("MYSQL_URI")
    engine = create_engine(MYSQL_URI)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM user_tables WHERE username=:username"), {"username": current_user})
        tables = [row[0] for row in result.fetchall()]
    return {"tables": tables}

# REMOVE /chat endpoint if present
# @app.post("/chat")
# def chat_endpoint(message: str):
#     ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)