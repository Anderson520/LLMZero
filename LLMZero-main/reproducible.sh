#!/bin/bash
export ALFWORLD_DATA="$PWD/environments/alfworld/"
alfworld-download
python reproducible.py