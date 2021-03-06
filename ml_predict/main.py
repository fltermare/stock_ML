from fastapi import FastAPI, BackgroundTasks
from query import predict_next_10days

app = FastAPI()


@app.post("/predict/{stock_code}")
async def do_predict(stock_code: str):
    # background_tasks.add_task(train, stock_code)
    res = predict_next_10days(stock_code)
    return {"predictions:": res}
    # return {"message": f"next price for {stock_code} is {res}"}
