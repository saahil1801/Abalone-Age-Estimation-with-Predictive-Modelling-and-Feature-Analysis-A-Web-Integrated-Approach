from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import lightgbm as lgb
import psycopg2
import json
from logging_config import setup_logging
import logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the pre-trained model
model = lgb.Booster(model_file='model.lgb')

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Database connection
def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",           # Database name
        user="postgres",           # Default user
        password="",               # No password
        host="localhost",
        port="5432"
    )

# Function to create the table if it doesn't exist
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        input_data JSONB,
        prediction FLOAT
    );
    '''
    try:
        cursor.execute(create_table_query)
        conn.commit()
        logger.info("Table 'predictions' created or already exists.")
    except Exception as e:
        logger.error(f"Error creating table: {e}")
    finally:
        cursor.close()
        conn.close()

# Call the create_table function when the app starts
@app.on_event("startup")
async def startup_event():
    create_table()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Serving root page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_age(
    request: Request,
    sex: int = Form(...),
    length: float = Form(...),
    diameter: float = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    shucked_weight: float = Form(...),
    viscera_weight: float = Form(...),
    shell_weight: float = Form(...)
):
    input_data = {
        "sex": sex,
        "length": length,
        "diameter": diameter,
        "height": height,
        "weight": weight,
        "shucked_weight": shucked_weight,
        "viscera_weight": viscera_weight,
        "shell_weight": shell_weight
    }
    logger.info(f"Received prediction request with data: {input_data}")
    
    # Convert the input data to a pandas DataFrame
    data = pd.DataFrame([input_data])
    
    # Make the prediction
    try:
        prediction = model.predict(data)
        prediction_value = prediction[0]
        logger.info(f"Prediction value: {prediction_value}")

        # Store the input data and prediction in the PostgreSQL database
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = '''
        INSERT INTO predictions (input_data, prediction)
        VALUES (%s, %s);
        '''
        cursor.execute(insert_query, (json.dumps(input_data), prediction_value))
        conn.commit()
        logger.info("Prediction stored in the database.")
    except Exception as e:
        logger.error(f"Error during prediction or storing data: {e}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()
    
    # Return the result template with the prediction
    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction_value})

@app.get("/view-predictions", response_class=HTMLResponse)
async def view_predictions(request: Request):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions;")
        rows = cursor.fetchall()
        
        predictions = []
        for row in rows:
            predictions.append({
                "id": row[0],
                "input_data": row[1],
                "prediction": row[2]
            })
        
        logger.info(f"Fetched {len(predictions)} predictions from the database.")
        return templates.TemplateResponse("predictions.html", {"request": request, "predictions": predictions})
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
