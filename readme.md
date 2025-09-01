# Text-to-SQL

This application transforms the way you interact with your data. Simply upload your own dataset (CSV/Excel), ask questions in natural language, and get instant SQL queries and results.

It's a user-friendly web interface that leverages a powerful yet efficient large language model (microsoft/Phi-3-mini-4k-instruct) to:

- Allow users to upload one or more custom datasets.

- Translate natural language questions into executable SQL queries.

- Run the queries against the user-provided data.

- Display the results instantly in a clean interface.

The entire backend, including the in-memory database and the AI model, runs on the server where the app is hosted.


# Features

- Custom Data Upload: Bring your own data! Upload CSV or Excel files, and the app will automatically create a database schema.

- Natural Language Queries: Ask questions in plain English instead of writing complex SQL.

- AI-Powered: Uses Microsoft's state-of-the-art Phi-3-mini model for high accuracy and efficienc

- Efficient: The model is quantized to 4-bit precision, making it suitable for deployment on free-tier servers like Hugging Face Spaces.

- Interactive UI: Built with Streamlit for a clean, simple, and responsive user experience.

- Schema Display: Users can view the database schema of their uploaded files to understand the available data.


# How to Run Locally
## Create a virtual environment (recommended):
```bash 
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

## Install the required libraries:
Make sure you have the app.py and requirements.txt files in the same directory.pip 
```bash
install -r requirements.txt
```

## Run the Streamlit application:
```bash
streamlit run app.py
```
The application will open in your web browser.

Note: The first time you run the app, it will download the AI model from Hugging Face, which is approximately 4-5 GB. This may take some time depending on your internet connection, but it only happens once.


# Deployment

Due to memory requirements, it is highly recommended to deploy this application to Hugging Face Spaces, which offers a generous free tier.