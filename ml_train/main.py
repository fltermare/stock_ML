from fastapi import FastAPI, BackgroundTasks
from train import train
import init

init.main()
app = FastAPI()

@app.post("/train/{stock_code}")
async def do_train(stock_code: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(train, stock_code)
    return {"message": f"start training model for {stock_code}"}
