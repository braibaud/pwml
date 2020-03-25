import pymssql as mssql
import pandas as pd


def get_data(proc_name, conn_params, proc_params):
    """
    Find the index of a value in a list of values
    :param proc_name: the fully qualified stored procedure name
    :param conn_params: a tuple containing the connection parameters (server, user, password, database)
    :param proc_params: a tuple containing the procedure parameters
    :return: a DataFrame of the first returned dataset
    """
    with connect(conn_params) as conn:
        with conn.cursor(as_dict=True) as cursor:
            cursor.callproc(
                proc_name,
                proc_params)

            cursor.nextset()
            results = cursor.fetchall()
            conn.commit()

            df = pd.DataFrame(results)

    return df

def connect(conn_params):
    return mssql.connect(*conn_params)

def execute(proc_name, conn_params, proc_params):
    """
    Find the index of a value in a list of values
    :param proc_name: the fully qualified stored procedure name
    :param conn_params: a tuple containing the connection parameters (server, user, password, database)
    :param proc_params: a tuple containing the procedure parameters
    """
    result = None

    with connect(conn_params) as conn:
        result = execute_base(
            proc_name,
            conn,
            proc_params)

    return result

def execute_base(proc_name, conn, proc_params):
    """
    Find the index of a value in a list of values
    :param proc_name: the fully qualified stored procedure name
    :param conn: a MS SQL database connection.
    :param proc_params: a tuple containing the procedure parameters
    """
    result = None

    with conn.cursor(as_dict=True) as cursor:
        cursor.callproc(
            proc_name,
            proc_params)

        if cursor.nextset():
            result = cursor.fetchone()

        conn.commit()

    return result