from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
from llama_index.legacy.query_engine import PandasQueryEngine
from fastapi.encoders import jsonable_encoder
from datetime import datetime, date
import json
import mysql.connector
from mysql.connector import Error
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "atliq_hospitality")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("Error: Missing OPENAI_API_KEY in .env file!")
if not EMAIL_SENDER or not EMAIL_PASSWORD:
    raise ValueError("Error: Missing EMAIL_SENDER or EMAIL_PASSWORD in .env file!")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle OPTIONS preflight requests
@app.options("/query/")
async def preflight():
    return {}

@app.options("/reviews/")
async def preflight_reviews():
    return {}

@app.options("/bookings/")
async def preflight_bookings():
    return {}

# Request Models
class QueryRequest(BaseModel):
    query: str

class ReviewRequest(BaseModel):
    name: str
    review: str
    rating: int

class BookingRequest(BaseModel):
    hotel_name: str
    customer_name: str
    contact_name: str
    contact_email: str
    check_in_date: date
    check_out_date: date
    time_in: str
    time_out: str
    num_guests: int
    guest_type: str

# Database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Error as e:
        raise RuntimeError(f"Error connecting to database: {e}")

# Send email
def send_booking_email(booking_data: dict):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = booking_data['contact_email']
        msg['Subject'] = 'Atliq Group Hotels - Booking Confirmation'

        body = f"""
        Dear {booking_data['contact_name']},

        Thank you for booking with Atliq Group Hotels! Here are your booking details:

        Hotel: {booking_data['hotel_name']}
        Customer Name: {booking_data['customer_name']}
        Contact Name: {booking_data['contact_name']}
        Contact Email: {booking_data['contact_email']}
        Check-in Date: {booking_data['check_in_date']}
        Check-out Date: {booking_data['check_out_date']}
        Time In: {booking_data['time_in']}
        Time Out: {booking_data['time_out']}
        Number of Guests: {booking_data['num_guests']}
        Guest Type: {booking_data['guest_type'].capitalize()}

        We look forward to welcoming you!

        Best regards,
        Atliq Group Hotels
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

# Load Excel File & Initialize PandasQueryEngine
EXCEL_FILE_PATH = "Atliq/data.xlsx"
try:
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name="fact_bookings")
    query_engine = PandasQueryEngine(df)
except Exception as e:
    raise RuntimeError(f"Error loading Excel file: {e}")

@app.get("/")
async def hello():
    return {"message": "Hello, World!"}

def ensure_jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        if obj.dtype.name.startswith('period'):
            return [x.to_timestamp().isoformat() if pd.notnull(x) else None for x in obj]
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return [ensure_jsonable(record) for record in obj.to_dict(orient="records")]
    elif isinstance(obj, pd.Period):
        return obj.to_timestamp().isoformat()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: ensure_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_jsonable(i) for i in obj]
    elif obj is None:
        return None
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

@app.post("/query/")
async def process_query(request: QueryRequest):
    try:
        response = query_engine.query(request.query)
        data_response = getattr(response, "response", "No data found.")
        clean_data_response = ensure_jsonable(data_response)

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

@app.post("/reviews/")
async def submit_review(request: ReviewRequest):
    try:
        if request.rating < 1 or request.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        connection = get_db_connection()
        cursor = connection.cursor()
        query = "INSERT INTO reviews (name, review, rating) VALUES (%s, %s, %s)"
        cursor.execute(query, (request.name, request.review, request.rating))
        connection.commit()
        cursor.close()
        connection.close()
        return {"message": "Review submitted successfully"}
    except Exception as e:
        print(f"Error submitting review: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reviews/")
async def get_reviews():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT name, review, rating, created_at FROM reviews ORDER BY created_at DESC LIMIT 4"
        cursor.execute(query)
        reviews = cursor.fetchall()
        cursor.close()
        connection.close()
        return {"reviews": reviews}
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bookings/")
async def submit_booking(request: BookingRequest):
    try:
        # Validate guest_type
        valid_guest_types = ['adult', 'child', 'mixed']
        if request.guest_type not in valid_guest_types:
            raise HTTPException(status_code=400, detail="Invalid guest type. Must be 'adult', 'child', or 'mixed'.")
        
        # Validate number of guests
        if request.num_guests < 1:
            raise HTTPException(status_code=400, detail="Number of guests must be at least 1.")
        
        # Validate dates
        if request.check_out_date <= request.check_in_date:
            raise HTTPException(status_code=400, detail="Check-out date must be after check-in date.")

        # Save to database
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
        INSERT INTO bookings (hotel_name, customer_name, contact_name, contact_email, check_in_date, check_out_date, time_in, time_out, num_guests, guest_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            request.hotel_name,
            request.customer_name,
            request.contact_name,
            request.contact_email,
            request.check_in_date,
            request.check_out_date,
            request.time_in,
            request.time_out,
            request.num_guests,
            request.guest_type
        ))
        connection.commit()
        cursor.close()
        connection.close()

        # Send confirmation email
        booking_data = request.dict()
        send_booking_email(booking_data)

        return {"message": "Booking submitted successfully. A confirmation email has been sent."}
    except Exception as e:
        print(f"Error submitting booking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    print("Starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    except KeyboardInterrupt:
        print("Server stopped manually")
    except asyncio.CancelledError:
        print("Async tasks cancelled during shutdown")