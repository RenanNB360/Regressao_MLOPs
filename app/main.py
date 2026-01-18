import io
import logging
import pandas as pd

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.services.model_service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    app.model_service = ModelService()
    logger.info("Model service loaded")

    yield

    logger.info("Shutting down application")

app = FastAPI(
    title="Regression Model API",
    version="1.0.0",
    description="FastAPI service for regression model inference",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty"
            )

        predictions = app.model_service.predict(df)

        return {
            "predictions": predictions.tolist()
        }
    
    except HTTPException as he:
        raise he

    except Exception as e:
        logger.error("Prediction error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
