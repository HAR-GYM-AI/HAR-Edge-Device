#!/usr/bin/env python3
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()
uri = os.getenv('MONGODB_URI')
print('MONGODB_URI:', uri)
try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print('Ping successful')
except Exception as e:
    print('Connection error:', e)
