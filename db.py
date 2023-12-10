from langchain.utilities import SQLDatabase
import sqlite3
import csv

# Create our database
db = SQLDatabase.from_uri("sqlite:///history.db")

# Function to connect to SQLite database
def get_db_connection():
    conn = sqlite3.connect('history.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # Create a table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            category TEXT,
            merchant TEXT,
            amount INTEGER
        );
    """)
    # Insert sample data from CSV file only if the table is empty
    if conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 0:
        with open('transactions.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            sample_data = list(reader)
            conn.executemany('INSERT INTO transactions (date, category, merchant, amount) VALUES (?, ?, ?, ?)', sample_data)
        
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

init_db()

# We can use db.table_info and check which sample rows are included:
print(db.table_info)

# Function to get table columns from SQLite database
def get_table_columns(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info({})".format(table_name))
    columns = cursor.fetchall()
    print(f"columns:{columns}")
    return [column[1] for column in columns]

table_name = 'transactions'
columns = get_table_columns(table_name)
