from dotenv import load_dotenv
from fastapi import FastAPI

from api.controller import router

# Load environment variables
load_dotenv()

app = FastAPI(title="ByteMe - ACE Alternative")
app.include_router(router=router)