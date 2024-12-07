#!/bin/bash

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install requirements
pip install -r Requirements.txt

# Prompt for Wandb login
echo "Please log in to Wandb:"
wandb login
echo "Setup complete. To activate the environment, run: source env/bin/activate"
