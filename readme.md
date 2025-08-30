# Text-to-SQL

This application allows you to interact with a database using natural language.

It's a user-friendly web interface that leverages a powerful, quantized large language model (HuggingFaceH4/zephyr-7b-beta) to:

- Translate your questions into executable SQL queries.

- Run the queries against a database.

- Display the results instantly.

The entire backend, including the database and the AI model, runs on the server where the app is hosted (either locally or on Streamlit Community Cloud).


# Features


Natural Language Queries: Ask questions in plain English instead of writing complex SQL.

- AI-Powered: Uses a state-of-the-art 7B parameter model for high accuracy.

- Efficient: The model is quantized to 4-bit precision, allowing it to run efficiently.

- Interactive UI: Built with Streamlit for a clean and simple user experience.

- Schema Display: Users can view the database schema to understand the available data.


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

You can easily deploy this application to the web for free using Streamlit Community Cloud.Upload the app.py, requirements.txt, and README.md files to a new GitHub repository.Sign in to Streamlit Community Cloud with your GitHub account.Click "New app" and select the repository you just created.Click "Deploy!"