from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np  
import openai
import os
from dotenv import load_dotenv
from llama_index.experimental.query_engine import PandasQueryEngine  
from fastapi.encoders import jsonable_encoder
from datetime import datetime, date
import json
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Error: Missing OPENAI_API_KEY in .env file!")

# Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Handle OPTIONS preflight request to fix 405 error
@app.options("/query/")
async def preflight():
    return {}

# Request Model
class QueryRequest(BaseModel):
    query: str

# Load Excel File & Initialize PandasQueryEngine
EXCEL_FILE_PATH = "Atliq/data.xlsx"
try:
    df = pd.read_excel(EXCEL_FILE_PATH,sheet_name="fact_bookings")
    query_engine = PandasQueryEngine(df)
except Exception as e:
    raise RuntimeError(f"Error loading Excel file: {e}")

@app.get("/")
async def hello():
    return {"message": "Hello, World!"}




def ensure_jsonable(obj):
    # Handle numpy types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        # Convert Series to list, handling PeriodDtype explicitly
        if obj.dtype.name.startswith('period'):
            return [x.to_timestamp().isoformat() if pd.notnull(x) else None for x in obj]
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to records, ensuring all columns are JSON-serializable
        return [ensure_jsonable(record) for record in obj.to_dict(orient="records")]
    elif isinstance(obj, pd.Period):
        # Handle individual Period objects
        return obj.to_timestamp().isoformat()
    elif isinstance(obj, (datetime, date)):
        # Handle datetime objects
        return obj.isoformat()
    # Handle nested structures
    elif isinstance(obj, dict):
        return {k: ensure_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_jsonable(i) for i in obj]
    # Handle None explicitly
    elif obj is None:
        return None
    # Fallback for other types (assumed to be JSON-serializable)
    try:
        json.dumps(obj)  # Test if the object is JSON-serializable
        return obj
    except (TypeError, OverflowError):
        # Convert non-serializable objects to string as a last resort
        return str(obj)


@app.post("/query/")
async def process_query(request: QueryRequest):
    try:
        # Query the Pandas DataFrame
        response = query_engine.query(request.query)
        data_response = getattr(response, "response", "No data found.")

        # Avoid eval; rely on ensure_jsonable for serialization
        clean_data_response = ensure_jsonable(data_response)

        # Query OpenAI GPT-4 API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        ai_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. The following query was run against a business dataset, and the result is included."},
                {"role": "user", "content": f"User asked: {request.query}\n\nQuery result: {clean_data_response}\n\nPlease explain the result in simple terms."}
            ]
        ).choices[0].message.content

        return jsonable_encoder({
            "data_response": clean_data_response,
            "ai_response": ai_response
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    print("Starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    except KeyboardInterrupt:
        print("Server stopped manually")
    except asyncio.CancelledError:
        print("Async tasks cancelled during shutdown")