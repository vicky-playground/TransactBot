from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('tracker.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    expenses = conn.execute('SELECT * FROM expense').fetchall()
    conn.close()
    return render_template('index.html', expenses=expenses)

@app.route('/add', methods=('GET', 'POST'))
def add():
    if request.method == 'POST':
        date = request.form['date']
        item = request.form['item']
        amount = request.form['amount']

        conn = get_db_connection()
        conn.execute('INSERT INTO expense (date, item, amount) VALUES (?, ?, ?)',
                     (date, item, amount))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))

    return render_template('add_expense.html')

if __name__ == '__main__':
    app.run(debug=True)
