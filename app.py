from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import lightgbm as lgb

app = FastAPI()

# Assume model is pre-trained and saved in 'your_model.lgb'
model = lgb.Booster(model_file='model.lgb')

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
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
    data = pd.DataFrame([input_data])
    prediction = model.predict(data)
    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction[0]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
