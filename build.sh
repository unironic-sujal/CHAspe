#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/uploads/original
mkdir -p static/uploads/processed
mkdir -p static/uploads/articles
mkdir -p detections

# Initialize the database
python init_db.py

# Make sure directories are accessible
chmod -R 755 static
chmod -R 755 instance 