from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
import pytz
from model import llm_hub, embeddings
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate

from datetime import datetime


from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
# connect to our database
db = SQLDatabase.from_uri("sqlite:///banking.db")

# We can use db.table_info and check which sample rows are included:
print(db.table_info)

# create the database chain
db_chain = SQLDatabaseChain.from_llm(llm_hub, db, verbose=True)


# Create db chain
QUERY = """
<s>[INST] 
Given an input inquiry related to banking transactions, create a syntactically correct SQLite-compatible SQL query to run. Then, execute this query against the 'transactions' table and provide the answer. 

Guidelines:
- If no specific date/time is mentioned, include all transactions.
- If a specific date/time period is mentioned, filter results using the current time zone: {time}.
- In the 'transactions' table, the 'amount' column indicates the monetary value of each transaction. Positive amounts in this column represent personal income or earnings, while negative amounts represent expenses or costs. The concept of 'balance' is calculated by summing all the amounts (both positive and negative) in the 'transactions' table (SELECT SUM(amount) AS Balance FROM transactions;).
- Always state expense amounts as positive figures to ensure clarity and consistency in communication.
- If the SQL query result is [(None,)], then the relevant amount is $0
- "Largest" or "smallest" should be understood as the transaction amount with the greatest absolute value (ORDER BY ABS(amount) DESC LIMIT 1;).
- When a query specifically mentions 'salary', you should focus on transactions where the description column includes the term 'salary' (WHERE description LIKE '%salary%').

<<SYS>>
Use the following format:
Query: Input inquiry
SQLQuery: SQL query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here with brief explanation

{inquiry}
<</SYS>>
[/INST]
"""

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('banking.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            description TEXT,
            amount INTEGER
        )
    """)
    # Insert sample data only if the table is empty
    if conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 0:
        sample_data = [
            ('2024-01-25', 'Salary', 2000),               # Income
            ('2023-12-25', 'Salary', 2000),               # Income
            ('2023-12-15', 'Groceries', -250), # Expense
            ('2023-12-01', 'Groceries', -150),   # Expense
            ('2023-03-15', 'Car Maintenance', -300),    # Expense
            ('2023-02-14', 'Valentine’s Gift', -100),   # Expense
            ('2023-01-01', 'New Year Party', -200)      # Expense            
        ]
        conn.executemany('INSERT INTO transactions (date, description, amount) VALUES (?, ?, ?)', sample_data)
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    conn = get_db_connection()
    transactions = conn.execute('SELECT id, * FROM transactions ORDER BY date DESC').fetchall()
    conn.close()
    return render_template('index.html', transactions=transactions)

@app.route('/inquiry', methods=['POST'])
def submit_inquiry():
    inquiry = request.form['inquiry']
    prompt = QUERY.format(time=datetime.now(pytz.timezone('America/New_York')), inquiry=inquiry)
    response = db_chain.run(prompt)
    print(f"answer: {response}")

    # Fetch transactions data again to pass to the template
    conn = get_db_connection()
    cursor = conn.execute('SELECT id, * FROM transactions ORDER BY date DESC')
    transactions = [dict(ix) for ix in cursor.fetchall()]
    conn.close()
    return render_template('index.html', inquiry=prompt, answer=response, transactions=transactions)

if __name__ == '__main__':
    app.run(debug=True)
