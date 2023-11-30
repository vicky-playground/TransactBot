from model import llm_hub, embeddings
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import sqlite3

# connect to our database
db = SQLDatabase.from_uri("sqlite:///history.db")

# create the database chain
db_chain = SQLDatabaseChain.from_llm(llm_hub, db, verbose=True)

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

# We can use db.table_info and check which sample rows are included:
print(db.table_info)
