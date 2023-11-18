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


loader = CSVLoader(file_path='data.csv')
pages = loader.load()
print("Number of pages loaded:", len(pages))  # Debugging print statement
print(pages)

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm_hub, verbose=True)

import sqlite3
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('tracker.db')
    conn.row_factory = sqlite3.Row
    # Check if the table exists, and create it if it doesn't
    conn.execute("""CREATE TABLE IF NOT EXISTS expense (
        date DATE,
        item TEXT,
        amount INTEGER
    )""")
    return conn


@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_db_connection()
    error_message = None

    if request.method == 'POST':
        date = request.form['date']
        item = request.form['item']
        amount = request.form['amount']

        conn.execute('INSERT INTO expense (date, item, amount) VALUES (?, ?, ?)', (date, item, amount))
        conn.commit()
        save_to_csv()
        return redirect(url_for('index'))

    # Fetch all expenses
    all_expenses = conn.execute('SELECT rowid, * FROM expense ORDER BY date DESC').fetchall()
    conn.close()
    return render_template('index.html', all_expenses=all_expenses)


@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    conn = get_db_connection()
    expense = conn.execute('SELECT * FROM expense WHERE rowid = ?', (id,)).fetchone()

    if request.method == 'POST':
        date = request.form['date']
        item = request.form['item']
        amount = request.form['amount']
        conn.execute('UPDATE expense SET date = ?, item = ?, amount = ? WHERE rowid = ?', (date, item, amount, id))
        conn.commit()
        save_to_csv()
        return redirect(url_for('index'))

    return render_template('edit.html', expense=expense)

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM expense WHERE rowid = ?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/records')
def records():
    conn = get_db_connection()
    expenses = conn.execute('SELECT rowid, * FROM expense ORDER BY date DESC').fetchall()

    # SQL to calculate total amount spent per month
    totals_by_month = conn.execute("""
        SELECT 
            strftime('%Y-%m', date) as month, 
            sum(amount) as total 
        FROM expense 
        GROUP BY strftime('%Y-%m', date)
        ORDER BY month DESC
    """).fetchall()

    conn.close()
    return render_template('records.html', expenses=expenses, totals_by_month=totals_by_month)


import csv

def save_to_csv():
    conn = get_db_connection()
    expenses = conn.execute('SELECT rowid, date, item, amount FROM expense ORDER BY date DESC').fetchall()
    conn.close()

    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Date', 'Item', 'Amount'])
        for expense in expenses:
            writer.writerow([expense['rowid'], expense['date'], expense['item'], expense['amount']])

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    if request.method == 'POST':
        question = request.form['question']
        if pages:
            # Initialize vector store only if pages are not empty
            vectordb = Chroma.from_documents(
                documents=pages,
                embedding=embeddings,
                persist_directory='docs/chroma'
            )
            vectordb.persist()

            """
            # Retrieval chain setup
            retriever = vectordb.as_retriever()
            qa = ConversationalRetrievalChain.from_llm(
                llm_hub,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(template)}
            )
            result = qa({"question": question})
            answer = result.get('answer', 'No answer found.')
            print(f"answer:{answer}")
            """

            # initialize vectordb retriever object
            qa = RetrievalQA.from_chain_type(
                llm=llm_hub,
                chain_type="stuff",
                retriever=vectordb.as_retriever(),
            )

            df = pd.read_csv("data.csv") # load employee_data.csv as dataframe
            python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

            # create calculator tool
            calculator = LLMMathChain.from_llm(llm=llm_hub, verbose=True)

            df_columns = df.columns.to_list() 

            # prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
            tools = [
                Tool(
                    name = "Expense Records",
                    func=qa.run,
                    description=f"""
                    Useful for when you need to answer questions about expense records. 
                    """
                ),                
                Tool(
                    name = "Expense Record Data",
                    func=python.run,
                    description=f"""
                    Useful for when you need to answer questions about expense records stored in pandas dataframe 'df'. 
                    Run python pandas operations on 'df' to help you get the right answer.
                     'df' has the following columns: {df_columns}      
                    """
                ),
                Tool(
                    name = "Calculator",
                    func=calculator.run,
                    description = f"""
                    Useful when you need to do math operations or arithmetic.
                    """
                )
            ]

            # change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
            agent_kwargs = {'prefix': f'You are friendly expense tracking assistant. You are tasked to assist the current user on questions related to expenses. You have access to the following tools:'}

            # initialize the LLM agent
            agent = initialize_agent(tools, 
                                    llm_hub, 
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                    verbose=True, 
                                    agent_kwargs=agent_kwargs
                                    )

            response = agent.run(question)
            print(f"response: {response}")

            return render_template('index.html', question=question, answer=response)
        else:
            return "No data loaded from CSV", 400

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
