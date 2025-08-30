import streamlit as st
import torch
from transformers import BitsAndBytesConfig, pipeline
import pandas as pd
from sqlalchemy import create_engine, inspect, text, Table, Column, Integer, String, Float, MetaData, ForeignKey
import re


# --- 1. Database Setup ---
@st.cache_resource
def setup_database():
    """Initializes an in-memory SQLite database and populates it with sample data."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()

    customers = Table(
        'customers', metadata_obj,
        Column('CustomerID', Integer, primary_key=True),
        Column('CustomerName', String(255)),
        Column('Region', String(255))
    )
    products = Table(
        'products', metadata_obj,
        Column('ProductID', Integer, primary_key=True),
        Column('ProductName', String(255)),
        Column('Category', String(255)),
        Column('Price', Float)
    )
    sales = Table(
        'sales', metadata_obj,
        Column('SaleID', Integer, primary_key=True),
        Column('ProductID', Integer, ForeignKey('products.ProductID')),
        Column('CustomerID', Integer, ForeignKey('customers.CustomerID')),
        Column('SaleAmount', Float)
    )
    metadata_obj.create_all(engine)

    # sample data
    with engine.connect() as conn:
        conn.execute(customers.insert(), [
            {'CustomerID': 1, 'CustomerName': 'Alice', 'Region': 'North'},
            {'CustomerID': 2, 'CustomerName': 'Bob', 'Region': 'South'},
            {'CustomerID': 3, 'CustomerName': 'Charlie', 'Region': 'North'}
        ])
        conn.execute(products.insert(), [
            {'ProductID': 1, 'ProductName': 'Laptop', 'Category': 'Electronics', 'Price': 1200.0},
            {'ProductID': 2, 'ProductName': 'Mouse', 'Category': 'Electronics', 'Price': 25.0},
            {'ProductID': 3, 'ProductName': 'Book', 'Category': 'Books', 'Price': 15.0}
        ])
        conn.execute(sales.insert(), [
            {'SaleID': 1, 'ProductID': 1, 'CustomerID': 1, 'SaleAmount': 1200.0},
            {'SaleID': 2, 'ProductID': 2, 'CustomerID': 1, 'SaleAmount': 25.0},
            {'SaleID': 3, 'ProductID': 3, 'CustomerID': 2, 'SaleAmount': 15.0},
            {'SaleID': 4, 'ProductID': 1, 'CustomerID': 3, 'SaleAmount': 1200.0},
        ])
        conn.commit()
    
    return engine

#--- 2. Load Model Pipeline ---
@st.cache_resource
def load_model_pipeline():
    """Loads the quantized 7B model and pipeline."""
    with st.spinner("Loading AI model... This may take a few moments."):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_pipeline = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",
            framework="pt",
            device_map="auto",
            model_kwargs={"quantization_config": quantization_config}
        )
    return model_pipeline

# --- 3. Helper Functions ---
def get_schema_string(db_engine):
    """Generates a CREATE TABLE string representation of the database schema."""
    inspector = inspect(db_engine)
    schema_str = ""
    for table_name in inspector.get_table_names():
        schema_str += f"CREATE TABLE {table_name} (\n"
        columns = inspector.get_columns(table_name)
        for i, column in enumerate(columns):
            col_name = column['name']
            col_type = str(column['type'])
            primary_key = " PRIMARY KEY" if column['primary_key'] else ""
            schema_str += f"  {col_name} {col_type}{primary_key}"
            if i < len(columns) - 1:
                schema_str += ",\n"
        schema_str += "\n);\n\n"
    return schema_str.strip()

def execute_query(sql_query, db_engine):
    """Executes the SQL query and returns the result as a pandas DataFrame."""
    try:
        with db_engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
        return result_df, None
    except Exception as e:
        return None, f"An error occurred: {e}"

# --- 4. Streamlit App ---
st.set_page_config(page_title="Text-to-SQL AI Assistant", layout="wide")
st.title("🤖 Text-to-SQL AI Assistant")

engine = setup_database()
pipe = load_model_pipeline()

db_schema = get_schema_string(engine)
with st.expander("View Database Schema"):
    st.code(db_schema, language="sql")

user_question = st.text_input("Enter your question about the data:", placeholder="e.g., Which product category has the highest total sales?")

if st.button("Generate and Execute SQL"):
    if user_question:
        with st.spinner("Generating SQL query..."):
            # System prompt with examples
            system_prompt = f"""### INSTRUCTIONS ###
You are a text-to-SQL engine.
- You must respond with ONLY a valid SQLite query.
- Your response must directly answer the user's question.
- DO NOT include explanations, comments, or markdown.
- DO NOT use a trailing semicolon.

### SCHEMA ###
{db_schema}

### EXAMPLES ###
[User Question]: How many customers are there?
[SQL Query]: SELECT COUNT(*) FROM customers

[User Question]: Which customers from the 'North' region bought a 'Laptop'?
[SQL Query]: SELECT c.CustomerName FROM customers c JOIN sales s ON c.CustomerID = s.CustomerID JOIN products p ON s.ProductID = p.ProductID WHERE c.Region = 'North' AND p.ProductName = 'Laptop'

[User Question]: Which product category has the highest total sales?
[SQL Query]: SELECT p.Category, SUM(s.SaleAmount) AS TotalSales FROM sales s JOIN products p ON s.ProductID = p.ProductID GROUP BY p.Category ORDER BY TotalSales DESC LIMIT 1
"""
            prompt_template = f"<|system|>\n{system_prompt}</s>\n<|user|>\n[User Question]: {user_question}</s>\n<|assistant|>\n[SQL Query]: "

            model_output = pipe(prompt_template, max_new_tokens=200, do_sample=False, truncation=True)
            
            full_output = model_output[0]['generated_text']
            assistant_response = full_output.split("[SQL Query]:")[-1].strip()
            select_match = re.search(r"SELECT", assistant_response, re.IGNORECASE)
            if select_match:
                generated_sql = assistant_response[select_match.start():].split('\n\n')[0]
                generated_sql = generated_sql.replace('```', '').rstrip('`').rstrip(';').strip()
            else:
                generated_sql = ""

        st.subheader("Generated SQL Query")
        if not generated_sql:
            st.error("The AI model failed to generate a valid SQL query. Please try rephrasing your question.")
        else:
            st.code(generated_sql, language="sql")
            
            st.subheader("Query Result")
            result_df, error = execute_query(generated_sql, engine)
            
            if error:
                st.error(error)
            elif result_df.empty:
                st.warning("Query executed successfully, but returned no results.")
            else:
                st.dataframe(result_df)
    else:
        st.warning("Please enter a question.")

