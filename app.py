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
    if request.method == 'POST':
        date = request.form['date']
        item = request.form['item']
        amount = request.form['amount']
        conn.execute('INSERT INTO expense (date, item, amount) VALUES (?, ?, ?)', (date, item, amount))
        conn.commit()
    expenses = conn.execute('SELECT rowid, * FROM expense ORDER BY date DESC LIMIT 5').fetchall()
    conn.close()
    return render_template('index.html', expenses=expenses)

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
    expenses = conn.execute('SELECT * FROM expense ORDER BY date DESC').fetchall()
    conn.close()
    return render_template('records.html', expenses=expenses)

if __name__ == '__main__':
    app.run(debug=True)
