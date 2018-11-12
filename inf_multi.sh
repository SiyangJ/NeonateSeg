#!/bin/bash

MODEL=10210052

echo 90000 100000 110000 | xargs -n1 -d" " -I {} _inf_iter $MODEL {}
