from flask import Flask, render_template, request
from datetime import datetime
import pytz
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate

from datetime import datetime

from db import *


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
