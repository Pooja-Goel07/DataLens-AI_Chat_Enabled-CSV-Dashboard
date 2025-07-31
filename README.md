# 📊 DataLens — AI Chat-Enabled-CSV-Dashboard

**DataLens** is an intelligent, web-based CSV explorer built for data analysts, students, and developers who want instant insights from structured data. Featuring a chatbot powered by Google Gemini (2.5 Flash), it allows users to upload CSV files, analyze them, and ask questions in natural language — all from a sleek and simple web interface.

---

## 🌐 Real-World Use Case
**Overwhelmed by messy CSVs?**

Manual data analysis is time-consuming and error-prone. DataLens automates insights and simplifies CSV interaction through chat-based querying and visual reports.
Use it to:
- Upload multiple CSV datasets
- Generate quick EDA reports (charts, stats, summaries)
- Ask questions like "Which column has the most nulls?" or "What's the average age?"

---

## 🧠 Core Features
### ✅ CSV Data Management

- Upload multiple CSVs via drag-and-drop UI
- Data stored securely in a MySQL database
- Easily switch between datasets

### 🔐 User Authentication
- JWT-based secure login/signup
- Passwords hashed with bcrypt
- Each user accesses only their uploaded files

### 📊 Automated Data Analysis
- Summary statistics
- Null value heatmaps
- Correlation matrices
- Histograms and categorical plots
- Outliers 

### 💬 Chatbot Data Querying
- Gemini 2.5 Flash API integration
- Asks & answers CSV-specific questions
- Learns context from current dataset

### 👥 Multi-User Support
- Each user sees only their uploaded files and reports
- Isolation through user IDs in database

---

## 🧱 App Architecture

| Layer         | Technology                         |
|---------------|-------------------------------------|
| Frontend      | HTML / CSS / JavaScript             |
| Backend       | FastAPI (Python 3.10), SQLModel     |
| Database      | MySQL                               |
| AI Model      |	LangChain + Gemini (Google Generative AI)         |
| Auth & Security| bcrypt, JWT Tokens, OAuth ready    |
|Charts         | 	Matplotlib, Seaborn                |
| Deployment    | Localhost                           |

> ⚠️ **Note:** The app uses **local MySQL** and **Gemini API**. Users must configure their own `.env` file and database.

---

## 🖼️ App Screens (UI Flow)
### 🤖 Chatbot Interface
- Query datasets in plain English  
- 📌 Example prompts:  
  - “Average income by region?”  
  - “Rows with salary > 100K and age < 30”  
  - “Plot age distribution”  

### 🏠 Home / Login
- Secure **Signup/Login** form with JWT-based authentication  
- Token stored in **local storage** for session management  
- Access restricted to **authenticated users only**  

### 📁 Dashboard
- **Upload** new CSV datasets via drag-and-drop or file browser  
- View **list of uploaded files** with timestamps  
- Easily **switch between datasets** for comparison and exploration  

### 📊 Data Report Summary

#### 📌 Summary
- Dataset name, upload time, rows, columns, missing values  
- Data quality score, outlier count 

#### 🧠 Key Findings
- AI-generated insights like:  
  - “Income is right-skewed with 8% outliers”  
  - “High correlation (r = 0.81) between Experience and Salary”  

#### 📑 Column-Level Analysis
- Data type, null count 
- Min/Max/Mean (numerical), uniqueness  

#### 🚨 Outlier Detection
- Z-score & IQR methods  
- Graph highlights + sample values  

#### 📈 Visualizations (6 Types)
- Histogram, Box Plot, Pie, Bar, Line (if datetime), Correlation Heatmap  

---

## 🚧 Known Limitations
- Currently runs only on `localhost`
- Requires manual `.env` setup
- Supports only CSV files (no Excel/JSON support yet)

## 🛡️ Security & Privacy
- ✅ Passwords are hashed using **bcrypt**
- ✅ JWT-based authentication for secure sessions
- ✅ Files not stored in plain-text
- ❌ No integration with third-party storage (AWS, GDrive, etc.)

---  

## 🚀 Get Started — DataLens Setup Guide

Follow these steps to run the project locally:
## 1. Clone the Repository
```bash
git clone https://github.com/Pooja-Goel07/DataLens-AI_Chat_Enabled-CSV-Dashboard
cd DataLens
```

## 2. Set Up Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # On Windows
# For Mac/Linux:
# source venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure Environment
Create a .env file in the backend/ folder with your:

- MySQL connection string
- GEMINI_API_KEY    #Gemini API key
- SECRET_KEY      # JWT Secret Key

## 4. Set Up MySQL Database
## 5. Run Backend Server
```BASH
uvicorn main:app --reload
```

## 6. Launch Frontend
Open `frontend/index.html` in your browser.





