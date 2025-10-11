#!/usr/bin/env python3
"""
MongoDB Connection Tester - Try Multiple Connection Methods
"""

from pymongo import MongoClient
import sys

# Common MongoDB connection strings
CONNECTION_STRINGS = [
    'mongodb://localhost:27017/',
    'mongodb://127.0.0.1:27017/',
    'mongodb+srv://cluster0.mongodb.net/',  # MongoDB Atlas cloud
]

print("="*60)
print("Testing MongoDB Connections...")
print("="*60)
print()

for idx, uri in enumerate(CONNECTION_STRINGS, 1):
    print(f"Test {idx}: Trying {uri}")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        print(f"  ✅ SUCCESS! Connected to {uri}")
        print()
        
        # List databases
        print("  Available databases:")
        dbs = client.list_database_names()
        for db_name in dbs:
            print(f"    - {db_name}")
            
        # Check for har_system database
        if 'har_system' in dbs:
            print()
            print("  ✅ Found 'har_system' database!")
            db = client['har_system']
            collections = db.list_collection_names()
            print(f"  Collections: {collections}")
            
            if 'training_data' in collections:
                collection = db['training_data']
                count = collection.count_documents({})
                print(f"  ✅ Found 'training_data' collection with {count} documents!")
                print()
                print("="*60)
                print("SUCCESS! Use this connection string:")
                print(f"  {uri}")
                print("="*60)
                
                # Show how to create .env file
                print()
                print("Create a .env file with:")
                print("-"*60)
                print(f"MONGODB_URI={uri}")
                print("MONGODB_DB_NAME=har_system")
                print("MONGODB_COLLECTION=training_data")
                print("-"*60)
                
                client.close()
                sys.exit(0)
        
        client.close()
        print()
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print()

print("="*60)
print("Could not connect to MongoDB with standard methods.")
print()
print("Please provide your MongoDB connection details:")
print("  1. Are you using MongoDB locally or MongoDB Atlas (cloud)?")
print("  2. What is your connection string?")
print("  3. What is your database name?")
print("="*60)
