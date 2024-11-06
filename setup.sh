#!/bin/bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
wandb login  # You may need to set up API key for Wandb here
