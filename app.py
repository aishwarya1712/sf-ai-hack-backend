import json

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from IndexManager import IndexManager

manager = IndexManager()
app = FastAPI()

# CORS setup
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataRequest(BaseModel):
    data: str

@app.get("/process")
async def process_data(data_request: str):
    response = manager.process(data_request)
    return {"response": response}

@app.get("/respond")
async def get_response():
    response = manager.get_response()
    # Convert non-serializable parts to serializable format
    return {"response": response}


@app.get("/", tags=["Root"])
async def read_root():
    # drive.list_files()
    # ingestion.get_response()
    return {"message": "This is working"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
