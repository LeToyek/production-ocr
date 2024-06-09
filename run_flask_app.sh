#!/bin/bash

# Set up environment variables
export FLASK_APP=app/main.py
export FLASK_ENV=development

# Create and activate the virtual environment
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check if Tesseract is installed and add to PATH if necessary
if ! command -v tesseract &> /dev/null; then
    echo "Tesseract-OCR is not installed. Please install it from https://digi.bib.uni-mannheim.de/tesseract/ and add it to your PATH."
    exit 1
fi

# Run the Flask application
flask run --host=0.0.0.0 --port=5000
