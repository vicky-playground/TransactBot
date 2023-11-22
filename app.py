# Assuming this function exists in model.py and utilizes the created CSV
from model import llm_hub, embeddings
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import os
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMMathChain

import sqlite3
from langchain.prompts import PromptTemplate

def get_db_connection():
    conn = sqlite3.connect('expenses.db')
    conn.row_factory = sqlite3.Row
    # Create the table with an auto-incrementing primary key
    conn.execute("""CREATE TABLE IF NOT EXISTS expenses (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE,
        item TEXT,
        amount INTEGER
    )""")
    conn.commit()
    return conn

"""
Building LLM Connected To Our Database
"""

conn = sqlite3.connect('records.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create the tasks table if it doesn't exist
conn.execute("""CREATE TABLE IF NOT EXISTS expenses (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE,
    item TEXT,
    amount INTEGER
)""")

# Insert sample tasks into the tasks table
data = [
    ('2023-11-11', 'Groceries', 150),
    ('2023-11-13', 'Utilities', 75),
    ('2023-11-15', 'Internet', 50),
    ('2023-11-18', 'Transport', 30),
    ('2023-11-20', 'Eating Out', 45),
    ('2023-11-22', 'Groceries', 160),
    ('2023-11-25', 'Gym Membership', 25),
    ('2023-12-18', 'Utilities', 80),
    ('2023-12-20', 'Clothing', 100),
    ('2023-12-23', 'Entertainment', 60),
    ('2023-12-25', 'Internet', 50),
    ('2023-12-28', 'Groceries', 120),
    ('2023-12-30', 'Phone Bill', 40)
]
# Ensure your database table structure matches these columns (date, item, amount)
cursor.executemany('INSERT INTO expenses (date, item, amount) VALUES (?, ?, ?)', data)

# Commit the changes and close the connection
conn.commit()
conn.close()

from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
# connect to our database
db = SQLDatabase.from_uri("sqlite:///records.db")

# We can use db.table_info and check which sample rows are included:
print(db.table_info)

# create the database chain
db_chain = SQLDatabaseChain.from_llm(llm_hub, db, verbose=True)

# Create db chain
QUERY = """
Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
Use the following format:

Query: {question}
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here


"""

question = "total amount?"
prompt = QUERY.format(question=question)
answer = db_chain.run(prompt)
print(f"answer: ")

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_db_connection()
    error_message = None

    if request.method == 'POST':
        date = request.form['date']
        item = request.form['item']
        amount = request.form['amount']

        # Insert new expense into the database
        conn.execute('INSERT INTO expenses (date, item, amount) VALUES (?, ?, ?)', (date, item, amount))
        conn.commit()
        save_to_csv()  # Update CSV file after adding new expense
        return redirect(url_for('index'))

    # Fetch all expenses from the database
    all_expenses = conn.execute('SELECT rowid, * FROM expenses ORDER BY date DESC').fetchall()
    for expense in all_expenses:
        print(expense['rowid'])  # This should print the rowid of each expense

    conn.close()

    return render_template('index.html', all_expenses=all_expenses)

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    conn = get_db_connection()
    expense = conn.execute('SELECT * FROM expenses WHERE rowid = ?', (id,)).fetchone()

    if request.method == 'POST':
        date = request.form['date']
        item = request.form['item']
        amount = request.form['amount']
        conn.execute('UPDATE expenses SET date = ?, item = ?, amount = ? WHERE rowid = ?', (date, item, amount, id))
        conn.commit()
        save_to_csv()
        return redirect(url_for('index'))

    conn.close()
    return render_template('edit.html', expense=expense)

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM expenses WHERE rowid = ?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/records')
def records():
    conn = get_db_connection()
    expenses = conn.execute('SELECT rowid, * FROM expenses ORDER BY date DESC').fetchall()

    # SQL to calculate total amount spent per month
    totals_by_month = conn.execute("""
        SELECT 
            strftime('%Y-%m', date) as month, 
            sum(amount) as total 
        FROM expenses
        GROUP BY strftime('%Y-%m', date)
        ORDER BY month DESC
    """).fetchall()

    conn.close()
    return render_template('records.html', expenses=expenses, totals_by_month=totals_by_month)


@app.route('/ask', methods=['POST'])
def ask_question():
    print(request.form)
    print(db.table_info)
    question = request.form['question']
    print(f"Question received: {question}")
    prompt = QUERY.format(question=question)
    response = db_chain.run(prompt)
    print(f"answer: {response}")
    return render_template('index.html', question=prompt, answer=response)

if __name__ == '__main__':
    app.run(debug=True)



