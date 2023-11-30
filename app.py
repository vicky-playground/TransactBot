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
db = SQLDatabase.from_uri("sqlite:///history.db")

# We can use db.table_info and check which sample rows are included:
print(db.table_info)

# create the database chain
db_chain = SQLDatabaseChain.from_llm(llm_hub, db, verbose=True)


# Create db chain
QUERY = """
<s>[INST] 
You are a powerful text-to-SQLite model. Your job is to answer questions about a database. You are given a question and context regarding the table of credit card transaction records which represents a person's expense records. You must run SQLite queries to the table to find an answer. Ensure your query does not include any non-SQLite syntax such as DATE_TRUNC or any use of backticks (`) or "```sql". Then, execute this query against the 'transactions' table and provide the answer. 
Provide strategies to manage expenses or tips to reduce the expenses in your answer as well. Compare the result with the spending in the previous months if any and include the insights in your answer.

Guidelines:
- Include all transactions if no specific date/time is mentioned. However, filter results to a specific date/time period using the current time zone: {time}. You should use ">=" or "<=" operators to filter the date or use "GROUP BY strftime('%m', date)" for grouping.  Assume the date format in the database is 'YYYY-MM-DD'.
- If the query result is [(None,)], run the SQLite query again to double check the answer. 
- If a specific category is mentioned in the inquiry, such as 'Groceries', 'Dining', or 'Utilities', use the "WHERE" condition in your SQL query to filter transactions by that category. For example, when asked for the average amount spent on 'Groceries', use "SELECT AVG(amount) FROM transactions WHERE category = 'Groceries'".
- If not asked for a specific category, yuou shouldn't filter any category out. On the other hand, you should use "where" condition to do the filtering. When asked for the average amount in a category, use the SQLite query (AVG(amount) WHERE category = 'category_name').
- When asked for \'highest\' or \'lowest\', use SQL function MAX() or MIN() respectively.

<<SYS>>
Use the following format to answer the inquiry:

Response: Result of the SQLite-compatible SQL query. If you know th transaction detailes such as the date, category, merchant, and amount, mention it in your answer to be more clear. - When asked for an \'overview\' of transactions, analyze and present the data in a way that highlights the proportions of different categories or types of transactions. For instance, calculate the percentage each category (like \'Groceries\', \'Dining\', \'Utilities\') contributes to the total transactions or total spending.
---------------------- line break
<br>
---------------------- line break
<br>
Explanation: Concise and succinct explanation on your thought process on how to get the final answer including the relevant transaction details such as the date, category, merchant, and amount.
---------------------- line break
<br>
---------------------- line break
<br>
Advice: Provide strategies to manage expenses or tips to reduce the expenses here.
<</SYS>>

{inquiry}
[/INST]
"""


app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('history.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            category TEXT,
            merchant TEXT,
            amount INTEGER
        );
    """)
    # Insert sample data only if the table is empty
    if conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 0:
        sample_data = [
            ('2023-12-05', 'Online Shopping', 'E-Store', 120),
            ('2023-12-02', 'Utilities', 'Internet Bill', 80),
            # Nov Transactions
            ('2023-11-29', 'Dining', 'Restaurant A', 85),
            ('2023-11-22', 'Groceries', 'Supermarket', 60),
            ('2023-11-15', 'Travel', 'Airline Ticket', 350),
            ('2023-11-12', 'Utilities', 'Electricity Bill', 75),
            ('2023-11-08', 'Online Shopping', 'Online Marketplace', 45),
            ('2023-11-05', 'Entertainment', 'Cinema', 30),
            # Transactions from earlier in the year
            ('2023-10-20', 'Dining', 'Cafe', 20),
            ('2023-10-15', 'Travel', 'Hotel Booking', 200),
            ('2023-10-10', 'Groceries', 'Local Market', 50),
            ('2023-10-06', 'Utilities', 'Water Bill', 40),
            ('2023-09-25', 'Online Shopping', 'Tech Store', 150),
            ('2023-09-15', 'Travel', 'Taxi Service', 25),
            ('2023-09-22', 'Dining', 'Restaurant B', 90),
            ('2023-09-10', 'Utilities', 'Electricity Bill', 100),
            ('2023-09-05', 'Groceries', 'Grocery Store', 80),
            ('2023-08-18', 'Utilities', 'Gas Bill', 60),
            ('2023-08-12', 'Entertainment', 'Streaming Service', 15),
            ('2023-08-06', 'Online Shopping', 'E-Store', 300),
            ('2023-07-30', 'Travel', 'Train Ticket', 40),
            ('2023-07-10', 'Dining', 'Fast Food', 25),
            ('2023-07-03', 'Utilities', 'Water Bill', 50)
        ]

        conn.executemany('INSERT INTO transactions (date, category, merchant, amount) VALUES (?, ?, ?, ?)', sample_data)
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
    # Determine if the inquiry is about spending advice
    if "reduce my spending" in inquiry:
        # Logic to handle advice-based inquiries
        response = handle_advice_inquiry(inquiry)
    else:
        # Handle regular transaction queries
        prompt = QUERY.format(time=datetime.now(pytz.timezone('America/New_York')), inquiry=inquiry)
        response = db_chain.run(prompt)
        #print(f"answer: {response}")

    # Fetch transactions data again to pass to the template
    conn = get_db_connection()
    cursor = conn.execute('SELECT id, * FROM transactions ORDER BY date DESC')
    transactions = [dict(ix) for ix in cursor.fetchall()]
    conn.close()

    # Replace newline characters with HTML break tags
    response = response.replace('\n', '<br>')
    
    return render_template('index.html', inquiry=prompt, answer=response, transactions=transactions)

if __name__ == '__main__':
    app.run(debug=True)
