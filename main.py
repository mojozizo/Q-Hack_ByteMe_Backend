from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.controller import router

# Load environment variables
load_dotenv()

app = FastAPI(title="ByteMe - ACE Alternative")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=router)