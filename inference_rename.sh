#!/bin/bash

cd inference$1

#ls | grep -P "model\.ckpt-\d\d\d\d\..*" | xargs -d"\n" rm
#ls | grep -P "model\.ckpt-\d\d\d\..*" | xargs -d"\n" rm
#ls | grep -P "model\.ckpt-\d*[123456789]0\..*" | xargs -d"\n" rm

MODEL_DIR=$(echo $PWD | grep -P -o "(?<=/nas/longleaf/home/siyangj/myNIRAL/model_)\d{8}")

ls | grep -P "\d(?=_niftynet_out\.nii\.gz)" -o | xargs -I {} -n1 -P1 -d"\n" bash -c "mv {}_niftynet_out.nii.gz {}_${MODEL_DIR}.nii.gz"

cd ../..


