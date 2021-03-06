import datetime
import json
import requests
import psycopg2
import pandas as pd

window_size = 20
DB_HOST = 'db'
TF_HOST = 'tfserving'

def db_conn():
    conn = psycopg2.connect(
        host=f"{DB_HOST}",
        # host='localhost',
        database='db',
        user='postgres',
        password='postgres'
    )

    cur = conn.cursor()
    cur.execute("SELECT version()")
    db_version = cur.fetchone()
    print(db_version)
    return conn

def rest_request(instance):
    url = f"http://{TF_HOST}:8501/v1/models/stocknet:predict"
#     payload = json.dumps({"instances": [matrix[:window_size].tolist(), 
#                                         matrix[1:window_size+1].tolist()]})
    payload = json.dumps({"instances": [instance]})
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=payload, headers=headers)
    print(response)
    return response


def get_lastest_data(stock_code, date, window_size, conn):
    para_p_sql = """
        SELECT * FROM (
            SELECT h.date, h.close, h.high, h.low, h.open, h.capacity, h.turnover, h.transactions, h.stock_code
            FROM history as h
            WHERE h.stock_code = %(stock_code)s and h.date <= %(date)s
            ORDER BY h.date DESC
            LIMIT %(window_size)s
        ) AS TEMP
        ORDER BY TEMP.date
    """
    df = pd.read_sql(para_p_sql, con=conn, params={'stock_code': stock_code, 'date': date, 'window_size': window_size})
    matrix = df[['close']].values
    matrix = matrix.tolist()
    time = df['date'].values

    return matrix, time


def predict_next_10days(stock_code):
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    conn = db_conn()
    matrix, time = get_lastest_data(stock_code, today, window_size, conn)
    # print(matrix)
    res_series = []
    for i in range(10):
        res = rest_request(matrix)
        predictions = json.loads(res.text)['predictions'][0]
        # print(predictions)
        res_series.append(predictions)
        matrix.append(predictions)
        matrix = matrix[1:]
    return res_series
