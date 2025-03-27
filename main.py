from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd  # Ensure pandas is imported

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/history")
def get_stock_history(ticker: str):
    try:
        stock_data = yf.download(ticker, period="1mo", interval="1d")

        # Check the multi-index structure and print the first few rows for debugging
        print(stock_data.head())  # Inspect the first few rows
        print(stock_data.columns)  # Check the column names

        # Extract 'Close' prices using multi-index columns
        if ('Close', ticker) in stock_data.columns:
            stock_data = stock_data[('Close', ticker)]  # Access the 'Close' column for the given ticker

        # Round the 'Close' prices to 2 decimals
        stock_data = stock_data.round(2)

        # Convert the date index and close prices to lists
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        prices = stock_data.tolist()

        return {"dates": dates, "prices": prices}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict/")
def predict(ticker: str = Form(...)):
    try:
        stock_data = yf.download(ticker, period="1y", interval="1d")
        
        if stock_data.empty:
            return JSONResponse(content={"detail": "Stock data not found for the given ticker"}, status_code=404)

        # Prepare data for prediction
        stock_data['Date'] = stock_data.index.map(lambda x: x.toordinal())
        X = stock_data[['Date']].values
        y = stock_data['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)

        # Predict next day's price
        future_day = np.array([[stock_data['Date'].max() + 1]])
        predicted_price = float(model.predict(future_day)[0])  # ✅ Convert NumPy float to Python float

        return JSONResponse(content={"predicted_price": predicted_price, "ticker": ticker})

    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)
