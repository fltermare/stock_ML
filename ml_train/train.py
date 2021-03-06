import os
import psycopg2
import pandas as pd
import tensorflow as tf
import time
# import matplotlib.pyplot as plt


window_size = 20
batch_size = 32
shuffle_buffer = 1000
ratio = 0.8
model_dir = '/models'

def db_conn():
    conn = psycopg2.connect(
        host='db',
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


def get_raw_dataset(conn, stock_code):
    para_p_sql = """
        SELECT h.date, h.close, h.high, h.low, h.open, h.capacity, h.turnover, h.transactions, h.stock_code
        FROM history as h
        WHERE h.stock_code = %(stock_code)s;
    """
    df = pd.read_sql(para_p_sql, con=conn, params={'stock_code': stock_code})

    matrix = df[['close']].values
    time = df['date'].values
    
    return matrix, time


def windowed_dataset_m(matrix, window_size=20, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(matrix)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], [window[-1:][0][0]]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_predict_m(model, matrix, window_size=20):
    ds = tf.data.Dataset.from_tensor_slices(matrix)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def get_dataset(matrix, time):
    train_idx = int(matrix.shape[0] * ratio)
    val_idx = (matrix.shape[0] - train_idx) // 2 + train_idx

    # training set
    time_train = time[:train_idx]
    x_train = matrix[:train_idx]

    # validation set
    time_valid = time[train_idx:val_idx]
    x_val = matrix[train_idx:val_idx]

    # test set
    time_test = time[val_idx:]
    x_test = matrix[val_idx:]

    print(matrix.shape)
    print(time_train.shape, time_valid.shape, time_test.shape)
    print(x_train.shape, x_val.shape, x_test.shape)

    train_ds = windowed_dataset_m(x_train, window_size, shuffle_buffer)
    valid_ds = windowed_dataset_m(x_val, window_size, shuffle_buffer)
    return train_ds, valid_ds


def raw_model_1d(window_size, dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=8,
                               kernel_size=5,
                               strides=1,
                               padding="causal",
                               activation="relu",
                               input_shape=[window_size, dim]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return model


def train_model(train_ds, valid_ds, dim):
    # dim = matrix.shape[1]
    model = raw_model_1d(window_size, dim)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    model.summary()

    # checkpoint_filepath = './tmp/checkpoint'
    checkpoint_filepath = model_dir + '/tmp/checkpoint'
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=50)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_mae',
        verbose=1,
        save_best_only=True,
    )

    history = model.fit(train_ds,
                        validation_data=valid_ds,
                        callbacks=[early_stop, model_checkpoint],
                        verbose=2,
                        epochs=500)

    model.load_weights(checkpoint_filepath)
    return model


def save_model(model):
    model_version = int(time.time())
    model_save_path = os.path.join(model_dir, "stocknet/{:d}/".format(model_version))
    tf.saved_model.save(model, model_save_path)


def train(stock_code: str):
    conn = db_conn()
    matrix, time = get_raw_dataset(conn, stock_code)
    train_ds, valid_ds = get_dataset(matrix, time)
    model = train_model(train_ds, valid_ds, dim=matrix.shape[1])
    save_model(model)


if __name__ == "__main__":
    print('new')
    train('0050')
