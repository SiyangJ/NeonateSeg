#!/bin/bash

ls | grep -P "(?<=model_)\d{8}" -o | xargs -d"\n" -n1 -I {} bash -c "./data_clean.sh {}"
