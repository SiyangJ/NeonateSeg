#!/bin/bash

ls | grep -P "model_\d{8}" | xargs -I {} -n1 -P1 -d"\n" bash -c "python csv_recorder.py {}/settings_training.txt"