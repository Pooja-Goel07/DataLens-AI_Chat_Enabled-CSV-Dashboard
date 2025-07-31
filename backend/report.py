# report.py - Fixed version with proper authentication
import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json
from upload import get_current_session, get_current_user

load_dotenv()

report_router = APIRouter()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

def get_dataframe_from_active_table():
    """Get DataFrame from the currently active table"""
    try:
        session = get_current_session()
        if not session["active_table"]:
            return None
        
        engine = create_engine(MYSQL_URI)
        query = f"SELECT * FROM {session['active_table']}"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return None

def convert_to_python_type(value):
    """Convert numpy types to native Python types"""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    else:
        return value

def generate_summary_stats(df):
    """Generate summary statistics for the dataset"""
    try:
        summary = {}
        
        # Basic info
        summary["basic_info"] = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "column_names": df.columns.tolist()
        }
        
        # Data types
        summary["data_types"] = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                summary["data_types"][col] = "numeric"
            elif dtype == 'object':
                summary["data_types"][col] = "text"
            elif 'datetime' in dtype:
                summary["data_types"][col] = "datetime"
            else:
                summary["data_types"][col] = dtype
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            summary["numeric_summary"] = {}
            for col in numeric_cols:
                if not df[col].isna().all():
                    summary["numeric_summary"][col] = {
                        "mean": convert_to_python_type(df[col].mean()),
                        "median": convert_to_python_type(df[col].median()),
                        "min": convert_to_python_type(df[col].min()),
                        "max": convert_to_python_type(df[col].max()),
                        "std": convert_to_python_type(df[col].std())
                    }
                    # Round if they are float values
                    for key, val in summary["numeric_summary"][col].items():
                        if isinstance(val, float):
                            summary["numeric_summary"][col][key] = round(val, 2)
                else:
                    summary["numeric_summary"][col] = {
                        "mean": None, "median": None, "min": None, "max": None, "std": None
                    }
        
        # Text columns summary
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            summary["text_summary"] = {}
            for col in text_cols:
                unique_count = int(df[col].nunique())
                most_common = convert_to_python_type(df[col].mode().iloc[0]) if not df[col].empty and len(df[col].mode()) > 0 else None
                sample_values = [convert_to_python_type(val) for val in df[col].dropna().head(3).tolist()]
                
                summary["text_summary"][col] = {
                    "unique_values": unique_count,
                    "most_common": most_common,
                    "sample_values": sample_values
                }
        
        return summary
    except Exception as e:
        return {"error": f"Error generating summary: {str(e)}"}

def detect_missing_values(df):
    """Detect missing values in the dataset"""
    try:
        missing_info = {}
        
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_info[col] = {
                "missing_count": missing_count,
                "missing_percentage": round(float(missing_percentage), 2),
                "has_missing": bool(missing_count > 0)
            }
        
        # Overall missing data summary
        total_missing = sum([info["missing_count"] for info in missing_info.values()])
        total_cells = len(df) * len(df.columns)
        overall_missing_percentage = (total_missing / total_cells) * 100
        
        return {
            "columns": missing_info,
            "summary": {
                "total_missing_values": int(total_missing),
                "total_cells": int(total_cells),
                "overall_missing_percentage": round(float(overall_missing_percentage), 2),
                "columns_with_missing": int(len([col for col, info in missing_info.items() if info["has_missing"]]))
            }
        }
    except Exception as e:
        return {"error": f"Error detecting missing values: {str(e)}"}

def detect_outliers(df):
    """Detect outliers using IQR method for numeric columns"""
    try:
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if df[col].isna().all():
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            # Convert to Python types
            sample_outliers = [convert_to_python_type(val) for val in outliers[col].head(5).tolist()] if outlier_count > 0 else []
            
            outliers_info[col] = {
                "outlier_count": int(outlier_count),
                "outlier_percentage": round(float(outlier_percentage), 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "has_outliers": bool(outlier_count > 0),
                "sample_outliers": sample_outliers
            }
        
        return outliers_info
    except Exception as e:
        return {"error": f"Error detecting outliers: {str(e)}"}

def generate_graph_suggestions(df):
    """Generate suggestions for graphs based on data types"""
    try:
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Histogram for numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            suggestions.append({
                "type": "histogram",
                "column": col,
                "title": f"Distribution of {col}",
                "description": f"Shows the frequency distribution of values in {col}"
            })
        
        # Bar chart for categorical columns with reasonable unique values
        for col in text_cols:
            unique_count = int(df[col].nunique())
            if 2 <= unique_count <= 20:  # Reasonable range for bar charts
                suggestions.append({
                    "type": "bar_chart",
                    "column": col,
                    "title": f"Count by {col}",
                    "description": f"Shows the frequency of different categories in {col}"
                })
        
        # Scatter plot for pairs of numeric columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter_plot",
                "columns": [numeric_cols[0], numeric_cols[1]],
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                "description": f"Shows the relationship between {numeric_cols[0]} and {numeric_cols[1]}"
            })
        
        # Box plot for numeric columns (good for outlier detection)
        for col in numeric_cols[:3]:  # Limit to first 3
            suggestions.append({
                "type": "box_plot",
                "column": col,
                "title": f"Box Plot of {col}",
                "description": f"Shows distribution and outliers in {col}"
            })
        
        return suggestions
    except Exception as e:
        return {"error": f"Error generating graph suggestions: {str(e)}"}

def generate_data_preview(df):
    """Generate a preview of the data"""
    try:
        # Convert DataFrame to records and ensure all values are JSON serializable
        def convert_records(records):
            converted = []
            for record in records:
                converted_record = {}
                for key, value in record.items():
                    converted_record[key] = convert_to_python_type(value)
                converted.append(converted_record)
            return converted
        
        preview = {
            "head": convert_records(df.head(5).to_dict('records')),
            "tail": convert_records(df.tail(3).to_dict('records')),
            "sample": convert_records(df.sample(min(3, len(df))).to_dict('records')) if len(df) > 8 else []
        }
        return preview
    except Exception as e:
        return {"error": f"Error generating data preview: {str(e)}"}

@report_router.get("/generate_report/")
def generate_report(current_user: str = Depends(get_current_user)):
    """Generate a comprehensive report for the active dataset"""
    try:
        session = get_current_session()
        
        if not session["active_table"]:
            raise HTTPException(status_code=400, detail="No active dataset. Please upload a CSV file first.")
        
        # Verify user owns the active table
        if session.get("current_user") != current_user:
            raise HTTPException(status_code=403, detail="You don't have access to the current active table.")
        
        # Get the DataFrame
        df = get_dataframe_from_active_table()
        if df is None:
            raise HTTPException(status_code=500, detail="Could not load data from active table.")
        
        # Generate all components of the report
        report = {
            "dataset_info": {
                "name": session["table_info"].get("original_name", "Unknown"),
                "file_name": session["table_info"].get("file_name", "Unknown"),
                "upload_time": session["upload_time"]
            },
            "summary": generate_summary_stats(df),
            "missing_values": detect_missing_values(df),
            "outliers": detect_outliers(df),
            "data_preview": generate_data_preview(df),
            "graph_suggestions": generate_graph_suggestions(df)
        }
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@report_router.get("/graph_data/{graph_type}/{column}")
def get_graph_data(graph_type: str, column: str, current_user: str = Depends(get_current_user)):
    """Get data for a specific graph"""
    try:
        session = get_current_session()
        
        # Verify user owns the active table
        if session.get("current_user") != current_user:
            raise HTTPException(status_code=403, detail="You don't have access to the current active table.")
        
        df = get_dataframe_from_active_table()
        if df is None:
            raise HTTPException(status_code=500, detail="Could not load data from active table.")
        
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in dataset.")
        
        if graph_type == "histogram":
            # Return data for histogram
            data = [convert_to_python_type(val) for val in df[column].dropna().tolist()]
            return {
                "type": "histogram",
                "data": data,
                "column": column,
                "bins": 20
            }
        
        elif graph_type == "bar_chart":
            # Return value counts for bar chart
            value_counts = df[column].value_counts().head(20)
            return {
                "type": "bar_chart",
                "labels": [convert_to_python_type(label) for label in value_counts.index.tolist()],
                "values": [int(val) for val in value_counts.values.tolist()],
                "column": column
            }
        
        elif graph_type == "box_plot":
            # Return data for box plot
            data = [convert_to_python_type(val) for val in df[column].dropna().tolist()]
            return {
                "type": "box_plot",
                "data": data,
                "column": column
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported graph type: {graph_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_graph_data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting graph data: {str(e)}")

@report_router.get("/scatter_data/{col1}/{col2}")
def get_scatter_data(col1: str, col2: str, current_user: str = Depends(get_current_user)):
    """Get data for scatter plot"""
    try:
        session = get_current_session()
        
        # Verify user owns the active table
        if session.get("current_user") != current_user:
            raise HTTPException(status_code=403, detail="You don't have access to the current active table.")
        
        df = get_dataframe_from_active_table()
        if df is None:
            raise HTTPException(status_code=500, detail="Could not load data from active table.")
        
        if col1 not in df.columns or col2 not in df.columns:
            raise HTTPException(status_code=400, detail="One or both columns not found in dataset.")
        
        # Get data for scatter plot, removing rows with NaN values
        clean_df = df[[col1, col2]].dropna()
        
        return {
            "type": "scatter_plot",
            "x_data": [convert_to_python_type(val) for val in clean_df[col1].tolist()],
            "y_data": [convert_to_python_type(val) for val in clean_df[col2].tolist()],
            "x_column": col1,
            "y_column": col2,
            "count": int(len(clean_df))
        }
    
    except Exception as e:
        print(f"Error in get_scatter_data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting scatter plot data: {str(e)}")
