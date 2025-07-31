# csv_analysis.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import warnings
import numpy as np
warnings.filterwarnings('ignore')

load_dotenv()

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def generate_graphs(df: pd.DataFrame, output_dir="graphs", table_name="dataset") -> tuple[list, list]:
    """Generate comprehensive visualizations for the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    graph_paths = []
    graph_descriptions = []

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # 1. Dataset Overview (Data types and missing values)
    if len(df.columns) > 0:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Data types chart
            dtype_counts = df.dtypes.value_counts()
            ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Data Types Distribution in {table_name}')
            
            # Missing values chart
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_data) > 0:
                ax2.bar(range(len(missing_data)), missing_data.values)
                ax2.set_xticks(range(len(missing_data)))
                ax2.set_xticklabels(missing_data.index, rotation=45, ha='right')
                ax2.set_title('Missing Values by Column')
                ax2.set_ylabel('Number of Missing Values')
            else:
                ax2.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Missing Values Status')
            
            plt.tight_layout()
            path = os.path.join(output_dir, f"{uuid4().hex}_overview.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_paths.append(path)
            graph_descriptions.append("Dataset Overview: Shows data types distribution and missing values analysis.")
        except Exception as e:
            print(f"Error generating overview chart: {e}")

    # 2. Correlation matrix for numeric data
    if len(numeric_cols) >= 2:
        try:
            plt.figure(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            mask = correlation_matrix.isnull()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8}, mask=mask)
            plt.title(f'Correlation Matrix - {table_name}')
            plt.tight_layout()
            path = os.path.join(output_dir, f"{uuid4().hex}_correlation.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_paths.append(path)
            graph_descriptions.append("Correlation Matrix: Shows relationships and correlations between numeric variables.")
        except Exception as e:
            print(f"Error generating correlation matrix: {e}")

    # 3. Distribution plots for numeric columns (max 6)
    for i, col in enumerate(numeric_cols[:6]):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram with KDE
            sns.histplot(df[col].dropna(), kde=True, ax=ax1, color="skyblue", alpha=0.7)
            ax1.set_title(f'Distribution of {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            
            # Box plot
            sns.boxplot(x=df[col].dropna(), ax=ax2, color="salmon")
            ax2.set_title(f'Box Plot of {col}')
            ax2.set_xlabel(col)
            
            plt.tight_layout()
            path = os.path.join(output_dir, f"{uuid4().hex}_{col.replace(' ', '_')}_dist.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_paths.append(path)
            graph_descriptions.append(f"Distribution Analysis of '{col}': Shows the data distribution, central tendency, and outliers.")
        except Exception as e:
            print(f"Error generating distribution plot for {col}: {e}")

    # 4. Top categorical distributions (max 4)
    for col in categorical_cols[:4]:
        try:
            # Only plot if reasonable number of categories
            unique_values = df[col].nunique()
            if 2 <= unique_values <= 20:
                plt.figure(figsize=(12, 6))
                
                # Get top 15 categories to avoid overcrowding
                top_categories = df[col].value_counts().head(15)
                
                # Create horizontal bar plot for better readability
                ax = top_categories.plot(kind='barh', color='skyblue')
                plt.title(f'Top Categories in {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                
                # Add value labels on bars
                for i, v in enumerate(top_categories.values):
                    ax.text(v + max(top_categories.values) * 0.01, i, str(v), 
                           va='center', fontweight='bold')
                
                plt.tight_layout()
                path = os.path.join(output_dir, f"{uuid4().hex}_{col.replace(' ', '_')}_categories.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_paths.append(path)
                graph_descriptions.append(f"Category Distribution of '{col}': Shows frequency of different categories.")
        except Exception as e:
            print(f"Error generating category plot for {col}: {e}")

    # 5. Numeric vs Categorical analysis (if both exist)
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        try:
            # Take first numeric and first categorical with reasonable categories
            num_col = numeric_cols[0]
            cat_col = None
            
            for col in categorical_cols:
                if 2 <= df[col].nunique() <= 10:
                    cat_col = col
                    break
            
            if cat_col:
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=df, x=cat_col, y=num_col, palette="Set2")
                plt.title(f'{num_col} by {cat_col}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                path = os.path.join(output_dir, f"{uuid4().hex}_numeric_vs_categorical.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_paths.append(path)
                graph_descriptions.append(f"Comparative Analysis: Shows how '{num_col}' varies across different '{cat_col}' categories.")
        except Exception as e:
            print(f"Error generating numeric vs categorical plot: {e}")

    # 6. Time series plot if datetime columns exist
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        try:
            date_col = datetime_cols[0]
            num_col = numeric_cols[0]
            
            # Ensure datetime column is properly formatted
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col, num_col])
            
            if len(df_temp) > 1:
                plt.figure(figsize=(14, 6))
                df_temp_sorted = df_temp.sort_values(date_col)
                plt.plot(df_temp_sorted[date_col], df_temp_sorted[num_col], marker='o', linewidth=2, markersize=4)
                plt.title(f'{num_col} over Time ({date_col})')
                plt.xlabel(date_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                path = os.path.join(output_dir, f"{uuid4().hex}_timeseries.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_paths.append(path)
                graph_descriptions.append(f"Time Series Analysis: Shows how '{num_col}' changes over time.")
        except Exception as e:
            print(f"Error generating time series plot: {e}")

    # 7. Scatter plots for pairs of numeric columns (max 3 pairs)
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols)-1)):
            try:
                x_col = numeric_cols[i]
                y_col = numeric_cols[i+1]
                path = os.path.join(output_dir, f"{uuid4().hex}_{x_col}_vs_{y_col}_scatter.png")
                plt.figure(figsize=(8, 5))
                plt.scatter(df[x_col], df[y_col], alpha=0.6, color="teal")
                plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_paths.append(path)
                graph_descriptions.append(f"Scatter plot of '{x_col}' vs '{y_col}': Shows relationship and possible correlation.")
            except Exception as e:
                print(f"Error generating scatter plot for {x_col} vs {y_col}: {e}")

    # 8. Pie chart for top category in categorical columns (max 2)
    for col in categorical_cols[:2]:
        try:
            if df[col].nunique() <= 10:
                path = os.path.join(output_dir, f"{uuid4().hex}_{col}_pie.png")
                plt.figure(figsize=(7, 7))
                df[col].value_counts().plot.pie(autopct='%1.1f%%', colors=plt.cm.Paired.colors)
                plt.title(f"Pie Chart of {col}")
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_paths.append(path)
                graph_descriptions.append(f"Pie chart of '{col}': Shows proportion of each category.")
        except Exception as e:
            print(f"Error generating pie chart for {col}: {e}")

    return graph_paths, graph_descriptions


def analyze_csv(file_path: str, table_name: str = "dataset"):
    """Comprehensive CSV analysis with enhanced statistics and insights"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading CSV: {str(e)}", "{}", []

    # Enhanced statistical summary
    try:
        # Basic info
        basic_stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "duplicate_rows": df.duplicated().sum()
        }

        # Column type analysis
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        
        column_analysis = {
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "datetime_columns": len(datetime_cols),
            "columns_with_nulls": df.isnull().any().sum(),
            "total_null_values": df.isnull().sum().sum()
        }

        # Generate detailed statistics for numeric columns
        numeric_stats = {}
        if numeric_cols:
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    numeric_stats[col] = {
                        "count": len(col_data),
                        "mean": col_data.mean(),
                        "median": col_data.median(),
                        "std": col_data.std(),
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "q25": col_data.quantile(0.25),
                        "q75": col_data.quantile(0.75),
                        "skewness": col_data.skew(),
                        "kurtosis": col_data.kurtosis(),
                        "nulls": df[col].isnull().sum(),
                        "null_percentage": (df[col].isnull().sum() / len(df)) * 100
                    }

        # Categorical column analysis
        categorical_stats = {}
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                categorical_stats[col] = {
                    "unique_values": col_data.nunique(),
                    "most_frequent": col_data.mode().iloc[0] if len(col_data.mode()) > 0 else "N/A",
                    "most_frequent_count": col_data.value_counts().iloc[0] if len(col_data) > 0 else 0,
                    "nulls": df[col].isnull().sum(),
                    "null_percentage": (df[col].isnull().sum() / len(df)) * 100
                }

        stats_summary = {
            "basic_info": basic_stats,
            "column_analysis": column_analysis,
            "numeric_statistics": numeric_stats,
            "categorical_statistics": categorical_stats
        }

    except Exception as e:
        stats_summary = {"error": f"Could not generate statistical summary: {str(e)}"}

    # Enhanced outlier detection
    outliers_dict = {}
    try:
        for column in numeric_cols:
            col_data = df[column].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                    
                    if not outliers.empty:
                        outliers_dict[column] = {
                            "count": len(outliers),
                            "percentage": (len(outliers) / len(col_data)) * 100,
                            "values": outliers.tolist()[:20],  # Limit to first 20 outliers
                            "bounds": {"lower": lower_bound, "upper": upper_bound},
                            "extreme_values": {
                                "min_outlier": outliers.min(),
                                "max_outlier": outliers.max()
                            }
                        }
    except Exception as e:
        outliers_dict = {"error": str(e)}

    # Generate graphs
    try:
        graph_paths, graph_descriptions = generate_graphs(df, table_name=table_name)
    except Exception as e:
        graph_paths, graph_descriptions = [], []
        print("Graph generation error:", e)

    # Enhanced AI summary with structured format
    try:
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.2, max_tokens=300)
        
        # Calculate key metrics
        missing_pct = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
        duplicate_pct = round((basic_stats.get('duplicate_rows', 0) / len(df)) * 100, 1) if len(df) > 0 else 0
        
        # Build concise but informative summary
        summary_text = f"""Dataset: {len(df)} rows, {len(df.columns)} columns
Types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical
Quality: {missing_pct}% missing data, {duplicate_pct}% duplicates
Outliers: {len(outliers_dict)} columns affected"""
        
        prompt = PromptTemplate(
            input_variables=["data"],
            template="""{data}

Provide analysis in this format:

## Dataset Overview
[2 sentences about size and structure]

## Data Quality
[Assessment of missing values, duplicates, outliers]

## Key Insights
[2-3 notable patterns or characteristics]

## Recommendations
[2 actionable suggestions]

Keep each section brief but informative."""
        )
        
        summary_response = (prompt | llm).invoke({"data": summary_text})
        ai_summary = summary_response.content.strip()
        
    except Exception as e:
        # Enhanced fallback summary with structure
        missing_pct = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
        duplicate_pct = round((basic_stats.get('duplicate_rows', 0) / len(df)) * 100, 1) if len(df) > 0 else 0
        
        ai_summary = f"""## Dataset Overview
This dataset contains {len(df):,} rows and {len(df.columns)} columns, providing a {'large' if len(df) > 10000 else 'medium' if len(df) > 1000 else 'small'}-scale dataset for analysis. The data includes {len(numeric_cols)} numeric variables and {len(categorical_cols)} categorical variables.

## Data Quality
Missing data accounts for {missing_pct}% of total values across the dataset. {'No duplicate rows detected.' if duplicate_pct == 0 else f'Duplicate rows represent {duplicate_pct}% of the data.'} {'Outliers detected in ' + str(len(outliers_dict)) + ' columns.' if len(outliers_dict) > 0 else 'No significant outliers detected.'}

## Key Insights
- Dataset structure: {'Primarily numeric data' if len(numeric_cols) > len(categorical_cols) else 'Mixed data types' if len(numeric_cols) > 0 else 'Primarily categorical data'}
- Memory footprint: {basic_stats.get('memory_usage', 0):.1f} MB
- Data completeness: {'High' if missing_pct < 5 else 'Moderate' if missing_pct < 20 else 'Low'} ({100-missing_pct:.1f}% complete)

## Recommendations
- {'Consider data cleaning for missing values' if missing_pct > 10 else 'Data quality appears good'}
- {'Investigate and handle outliers before analysis' if len(outliers_dict) > 0 else 'Proceed with standard analysis techniques'}"""

    return ai_summary, json.dumps(convert_numpy(outliers_dict), indent=2), graph_paths


def quick_analyze_dataframe(df: pd.DataFrame, table_name: str = "dataset"):
    """Quick analysis for DataFrames already in memory (used by main app)"""
    try:
        # Create temporary CSV for full analysis
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{table_name}_temp.csv")
        df.to_csv(temp_file, index=False)
        
        # Run full analysis
        summary, outliers_json, graph_paths = analyze_csv(temp_file, table_name)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return summary, outliers_json, graph_paths
        
    except Exception as e:
        return f"Quick analysis failed: {str(e)}", "{}", []


def generate_comparison_report(df1: pd.DataFrame, df2: pd.DataFrame, name1: str = "Dataset 1", name2: str = "Dataset 2"):
    """Generate a comparison report between two datasets"""
    try:
        comparison = {
            "basic_comparison": {
                name1: {"rows": len(df1), "columns": len(df1.columns)},
                name2: {"rows": len(df2), "columns": len(df2.columns)}
            },
            "column_comparison": {
                "common_columns": list(set(df1.columns) & set(df2.columns)),
                "unique_to_first": list(set(df1.columns) - set(df2.columns)),
                "unique_to_second": list(set(df2.columns) - set(df1.columns))
            }
        }
        
        # Generate comparison visualization if datasets have common columns
        common_cols = comparison["column_comparison"]["common_columns"]
        numeric_common = [col for col in common_cols if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']]
        
        graph_paths = []
        if len(numeric_common) > 0:
            try:
                fig, axes = plt.subplots(1, min(3, len(numeric_common)), figsize=(15, 5))
                if len(numeric_common) == 1:
                    axes = [axes]
                
                for i, col in enumerate(numeric_common[:3]):
                    ax = axes[i] if len(numeric_common) > 1 else axes[0]
                    ax.hist(df1[col].dropna(), alpha=0.7, label=name1, bins=20)
                    ax.hist(df2[col].dropna(), alpha=0.7, label=name2, bins=20)
                    ax.set_title(f'Comparison: {col}')
                    ax.legend()
                
                plt.tight_layout()
                path = os.path.join("graphs", f"{uuid4().hex}_comparison.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_paths.append(path)
            except Exception as e:
                print(f"Error generating comparison plot: {e}")
        
        return comparison, graph_paths
        
    except Exception as e:
        return {"error": str(e)}, []


def convert_numpy(obj):
    """Convert numpy objects to native Python types for JSON serialization"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


def set_active_table(table_name):
    """When user selects a new table"""
    global active_table, chat_history
    active_table = table_name
    chat_history = []  # Clear chat history