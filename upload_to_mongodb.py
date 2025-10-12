#!/usr/bin/env python3
"""
Upload JSON session data to MongoDB
Checks for existing sessions and uploads only new ones
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'har_system')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'training_data')

def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI)
        # Test the connection
        client.admin.command('ping')
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        logger.info(f"Successfully connected to MongoDB: {MONGODB_DB_NAME}.{MONGODB_COLLECTION}")
        return client, db, collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None, None, None

def is_session_uploaded(collection, session_data):
    """Check if session is already uploaded by start_time_utc_ms"""
    start_time = session_data.get('session_metadata', {}).get('start_time_utc_ms')
    if start_time is None:
        return False

    # Check if document exists with same start time
    existing = collection.find_one({
        'session_metadata.start_time_utc_ms': start_time
    })
    return existing is not None

def upload_session_to_mongodb(collection, session_data):
    """Upload a single session to MongoDB"""
    try:
        # Add MongoDB-specific metadata
        session_data['_created_at'] = datetime.now()
        session_data['_data_source'] = 'batch_upload'

        # Insert the document
        result = collection.insert_one(session_data)
        logger.info(f"Session uploaded to MongoDB with ID: {result.inserted_id}")
        return True

    except Exception as e:
        logger.error(f"Error uploading session to MongoDB: {e}")
        return False

def main():
    """Main upload function"""
    # Initialize MongoDB
    client, db, collection = init_mongodb()
    if collection is None:
        logger.error("Cannot proceed without MongoDB connection")
        return

    # Get data directory
    data_dir = Path("collected_data")
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return

    # Get all JSON files
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        logger.info("No JSON files found in collected_data directory")
        return

    logger.info(f"Found {len(json_files)} JSON files to check")

    uploaded_count = 0
    skipped_count = 0
    error_count = 0

    for json_file in sorted(json_files):
        try:
            logger.info(f"Processing {json_file.name}...")

            # Load JSON data
            with open(json_file, 'r') as f:
                session_data = json.load(f)

            # Check if already uploaded
            if is_session_uploaded(collection, session_data):
                logger.info(f"  -> Already uploaded, skipping")
                skipped_count += 1
                continue

            # Upload to MongoDB
            if upload_session_to_mongodb(collection, session_data):
                uploaded_count += 1
                logger.info(f"  -> Successfully uploaded")
            else:
                error_count += 1
                logger.error(f"  -> Failed to upload")

        except json.JSONDecodeError as e:
            logger.error(f"  -> Invalid JSON in {json_file.name}: {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"  -> Error processing {json_file.name}: {e}")
            error_count += 1

    # Summary
    logger.info("\n" + "="*50)
    logger.info("UPLOAD SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files processed: {len(json_files)}")
    logger.info(f"Successfully uploaded: {uploaded_count}")
    logger.info(f"Already uploaded: {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("="*50)

    # Close connection
    if client:
        client.close()

if __name__ == "__main__":
    main()