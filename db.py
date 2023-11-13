import sqlite3

connection = sqlite3.connect("tracker.db")
cursor = connection.cursor()
cursor.execute("CREATE TABLE expense (date DATE, item TEXT, amount INTEGER)")
cursor.execute("INSERT INTO expense VALUES(\"2023-11-13\", \"drink\", 300)")


rows = cursor.execute("SELECT * FROM expense").fetchall()
print(rows)
