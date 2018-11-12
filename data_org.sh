#!/bin/bash

OLD_DIR=/proj/NIRAL/users/siyangj/NeonateMRISeg/TrainigData/
NEW_DIR=/proj/NIRAL/users/siyangj/myData/

for i in 1 2 3 4 5 6 7 8
do
cp $OLD_DIR$i/$i-seg.nrrd $NEW_DIR
cp $OLD_DIR$i-flip/$i-flip-seg.nrrd $NEW_DIR
cp $OLD_DIR$i/$i-T1-stripped.nrrd $NEW_DIR
cp $OLD_DIR$i/$i-T2-stripped.nrrd $NEW_DIR
cp $OLD_DIR$i-flip/$i-flip-T1-stripped.nrrd $NEW_DIR
cp $OLD_DIR$i-flip/$i-flip-T2-stripped.nrrd $NEW_DIR
done
