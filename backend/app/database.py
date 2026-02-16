import os
import motor.motor_asyncio
from dotenv import load_dotenv

load_dotenv()
client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGODB_URL"))
db = client.aegis_db
chat_collection = db.threat_logs