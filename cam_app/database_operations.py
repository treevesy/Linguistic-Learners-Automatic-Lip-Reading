import sqlite3, datetime
def create_table(dbname, tablename, text):
    script_sql = f"""CREATE TABLE IF NOT EXISTS {str(tablename)} ({text});"""
    try:
        sqliteConnection = sqlite3.connect(str(dbname))
        cursor = sqliteConnection.cursor()
        print("Successfully Connected to SQLite")
        print(script_sql)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print("table names: ", cursor.fetchall())
        cursor.execute(script_sql)

    except sqlite3.Error as error:
        print("Error while executing sqlite script", error)

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")

def add_data_to_table(dbname, tablename, table_metadata, data):
    strdata=tuple(data)
    strmeta=tuple(table_metadata)
    print(strmeta)
    print(strdata)
    script_sql_add = f"""INSERT INTO "main"." {tablename} " {str(strmeta)} VALUES {strdata};"""

    try:
        sqliteConnection = sqlite3.connect(str(dbname))
        cursor = sqliteConnection.cursor()
        print("Successfully Connected to SQLite")
        print(script_sql_add)
        cursor.execute(script_sql_add)
        sqliteConnection.commit()


    except sqlite3.Error as error:
        print("Error while executing sqlite script", error)

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")

def view_table(dbname, tablename):
    try:
        sqliteConnection = sqlite3.connect(str(dbname))
        cursor = sqliteConnection.cursor()
        print("Successfully Connected to SQLite")

        cursor.execute(f"SELECT * FROM '{tablename}';")
        details = cursor.fetchall()
        print("result: ", details)


    except sqlite3.Error as error:
        print("Error while executing sqlite script", error)

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")

def update_data_table(dbname, tablename, text):
    try:
        sqliteConnection = sqlite3.connect(str(dbname))
        cursor = sqliteConnection.cursor()
        print("Successfully Connected to SQLite")
        script_sql_update = f"""UPDATE {tablename} SET {text};"""
        cursor.execute(script_sql_update)
        sqliteConnection.commit()


    except sqlite3.Error as error:
        print("Error while executing sqlite script", error)

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")
