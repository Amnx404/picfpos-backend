from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from uuid import uuid4
import inference
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os

# Add an import for Float
from sqlalchemy import Float
import shutil
import graph

# Database and other configurations
DATABASE_URL = "sqlite:///./test7.db"
FRAMES_DIR = "frames"
MAX_FILES_PER_SESSION = 50

# Create engine and tables
database = Database(DATABASE_URL)
metadata = MetaData()
engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

# FastAPI app initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Modify the predictions table definition
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, index=True, autoincrement=True),
    Column("session_id", String),
    Column("prediction", String),
    Column("alpha", Float),  # New column for alpha value
    Column("final_prediction", String)  # New column for final prediction
)


@app.on_event("startup")
async def startup():
    await database.connect()
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    #if table exists connect, else create table
    predictions.create(bind=engine, checkfirst=True)


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Other imports and setup...

@app.get("/")
async def root():
    return {"message": "Hello World"}

THRESHOLD = 15  # Adjust this threshold as needed

@app.post("/predict")
async def predict(file: UploadFile = File(...), session_id: str = Form(None), alpha: float = Form(...)):
    try:
        print("all Items in form", file, session_id, alpha)
        if not session_id:
            session_id = str(uuid4())

        contents = await file.read()

        # Manage file creation and deletion
        prediction, file_path = await handle_file_and_predict(session_id, contents)

        # Save prediction to the database with the alpha value
        query = predictions.insert().values(session_id=session_id, prediction=prediction, alpha=alpha)
        await database.execute(query)

        # Optionally, delete the temporary file
        if file_path:
            os.remove(file_path)

        # Calculate the final prediction
        final_prediction = await calculate_final_prediction(session_id)
        print("response", prediction, final_prediction, session_id)
        return {"prediction": prediction, "final_prediction": final_prediction, "session_id": session_id}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


from fastapi import HTTPException
from pydantic import BaseModel

class NavigateRequest(BaseModel):
    session_id: str
    end: str
    alpha: float


@app.post("/navigate")
async def navigate(request: NavigateRequest):
    try:
        print("navigate", request.session_id, request.end, request.alpha)
        # Fetch final prediction for the session
        final_query = predictions.select().where(predictions.c.session_id == request.session_id).order_by(-predictions.c.id).limit(1)
        
        final_row = await database.fetch_one(final_query)
        current_position = final_row["final_prediction"]
        if current_position is None:
            raise HTTPException(status_code=400, detail="No final prediction available")

        path, directions = graph.navigate(current_position, request.end, request.alpha)

        return {"path": path, "directions": directions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_locations")
async def get_locations():
    return ['3D', 'C0', 'C1', 'C10', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9','E1', 'E2', 'ENT', 'EX', 'EXB', 'I', 'M', 'MEET', 'MOD', 'R', 'SEW',     'SV', 'TECH', 'WM1', 'WM2']


async def calculate_final_prediction(session_id):
    # Fetch the last 30 predictions for the session
    query = predictions.select().where(predictions.c.session_id == session_id).order_by(-predictions.c.id).limit(30)
    last_30_rows = await database.fetch_all(query)

    # Count the occurrences of each prediction
    prediction_counts = Counter(row['prediction'] for row in last_30_rows)

    # Find the most frequent prediction if it meets the threshold
    most_common_prediction, count = prediction_counts.most_common(1)[0]
    if count >= THRESHOLD:
        # Update the final prediction for these rows
        update_query = predictions.update().where(predictions.c.session_id == session_id).values(final_prediction=most_common_prediction)
        await database.execute(update_query)
        return most_common_prediction
    else:
        return None

# Additional functions and routes...


async def handle_file_and_predict(session_id, contents):
    # Calculate total locations from database for the session ID
    query = predictions.select().where(predictions.c.session_id == session_id)
    rows = await database.fetch_all(query)
    total_locations = len(rows)

    # Check if file count exceeds the limit
    if total_locations >= MAX_FILES_PER_SESSION:
        await delete_oldest_file(session_id)

    # Create a temporary file for inference
    with NamedTemporaryFile(delete=False, dir=FRAMES_DIR, suffix=".jpg") as temp_file:
        temp_file.write(contents)
        temp_file_path = temp_file.name

    # Perform prediction
    prediction = inference.do_inference(temp_file_path)

    return prediction, temp_file_path

async def delete_oldest_file(session_id):
    session_files = [f for f in os.listdir(FRAMES_DIR) if f.startswith(session_id)]
    if session_files:
        oldest_file = min(session_files, key=lambda x: os.path.getctime(os.path.join(FRAMES_DIR, x)))
        os.remove(os.path.join(FRAMES_DIR, oldest_file))

# Additional routes and logic...


# Additional routes and logic...
