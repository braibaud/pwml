import pymssql as mssql
import pandas as pd


def connect(conn_params):
    return mssql.connect(*conn_params)

def execute(proc_name, conn_params, proc_params):
    """
    Executes a stored-procedure and returns a list of dictionaries (as a representation of the resultset).
    :param proc_name: the fully qualified stored procedure name
    :param conn_params: a tuple containing the connection parameters (server, user, password, database)
    :param proc_params: a tuple containing the procedure parameters
    :return: a DataFrame of the first returned dataset
    """
    result = None

    with connect(conn_params) as conn:
        with conn.cursor(as_dict=True) as cursor:
            cursor.callproc(
                proc_name,
                proc_params)

            if cursor.nextset():
                result = cursor.fetchall()
            
            conn.commit()

    return result
