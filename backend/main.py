# main.py - Fixed version with proper error handling and report router
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from upload import upload_csv
from query import ask_question
from report import report_router  # Add this import
from csv_analysis import analyze_csv, generate_graphs
import pandas as pd
import os
import json
from datetime import datetime, timedelta
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
    allow_origins=["*"],
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
app.include_router(report_router, prefix="/api")  # Add this line

# Authentication setup
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

if not SECRET_KEY:
    raise ValueError("SECRET_KEY not found in environment variables.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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
    
    MYSQL_URI = os.getenv("MYSQL_URI")
    engine = create_engine(MYSQL_URI)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT username FROM users WHERE username=:username"), {"username": username})
        user = result.fetchone()
        if not user:
            raise credentials_exception
    return username

# User model
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str

# Create database tables
MYSQL_URI = os.getenv("MYSQL_URI")
engine = sqlmodel_create_engine(MYSQL_URI, echo=True)
SQLModel.metadata.create_all(engine)

class SignupRequest(BaseModel):
    username: str
    password: str

@app.post("/api/signup/")
async def signup(payload: SignupRequest):
    """Register a new user"""
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
    """User login endpoint"""
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

@app.get("/api/list_tables/")
async def list_tables(current_user: str = Depends(get_current_user)):
    """List tables for authenticated user"""
    try:
        from upload import get_current_session
        session = get_current_session()
        
        engine = create_engine(MYSQL_URI)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name, original_name FROM user_tables 
                WHERE username = :username
                ORDER BY upload_time DESC
            """), {"username": current_user})
            
            tables = [{"table_name": row[0], "original_name": row[1]} for row in result.fetchall()]
            
            return {
                "tables": [t["table_name"] for t in tables],
                "table_details": tables,
                "active_table": session.get("active_table"),
                "user": current_user
            }
    except Exception as e:
        print(f"Error in list_tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/api/generate_report/")
async def generate_report(table_name: str = None, current_user: str = Depends(get_current_user)):
    """Generate report for user's table"""
    try:
        from upload import get_current_session
        session = get_current_session()
        
        target_table = table_name if table_name else session.get("active_table")
        
        if not target_table:
            raise HTTPException(
                status_code=400,
                detail="No active table found. Please upload a CSV file first."
            )
        
        # Verify user owns this table
        engine = create_engine(MYSQL_URI)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT original_name, file_name FROM user_tables 
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": target_table})
            
            table_record = result.fetchone()
            if not table_record:
                raise HTTPException(status_code=403, detail="Table not found or access denied")
            
            # Export table data to DataFrame
            df = pd.read_sql(f"SELECT * FROM {target_table}", conn)
        
        # Create temporary file for analysis
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{target_table}_temp.csv")
        df.to_csv(temp_file, index=False)
        
        # Generate analysis
        summary, outliers_json, graph_paths = analyze_csv(temp_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Prepare response
        graph_urls = [f"/graphs/{os.path.basename(path)}" for path in graph_paths]
        
        return {
            "table_name": target_table,
            "original_name": table_record[0],
            "file_name": table_record[1],
            "summary": summary,
            "outliers": json.loads(outliers_json) if outliers_json else {},
            "graph_urls": graph_urls,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in generate_report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/api/switch_table/")
async def switch_active_table(request: dict = Body(...), current_user: str = Depends(get_current_user)):
    """Switch active table for authenticated user"""
    table_name = request.get("table_name")
    if not table_name:
        raise HTTPException(status_code=400, detail="table_name is required")
    
    try:
        from upload import get_current_session
        
        # Verify user owns this table
        engine = create_engine(MYSQL_URI)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT original_name FROM user_tables 
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": table_name})
            
            table_record = result.fetchone()
            if not table_record:
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found or access denied")
            
            # Get table info
            columns_result = conn.execute(text(f"DESCRIBE {table_name}"))
            columns = [row[0] for row in columns_result.fetchall()]
            
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = count_result.fetchone()[0]
        
        # Update session
        session = get_current_session()
        session["active_table"] = table_name
        session["current_user"] = current_user
        session["table_info"] = {
            "original_name": table_record[0],
            "safe_name": table_name,
            "columns": columns,
            "rows": row_count,
            "switched_at": datetime.now().isoformat()
        }
        
        return {
            "message": f"Switched to table '{table_record[0]}'",
            "active_table": table_name,
            "table_info": session["table_info"],
            "chat_history": []
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in switch_active_table: {e}")
        raise HTTPException(status_code=500, detail=f"Error switching table: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
