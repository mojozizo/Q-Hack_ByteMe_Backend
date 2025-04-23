from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
from api.controller import router

app = FastAPI(title="ByteMe - ACE Alternative")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=router)