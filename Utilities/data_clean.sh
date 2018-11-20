#!/bin/bash

cd model_$1/models/

#ls | grep -P "model\.ckpt-\d\d\d\d\..*" | xargs -d"\n" rm
#ls | grep -P "model\.ckpt-\d\d\d\..*" | xargs -d"\n" rm
#ls | grep -P "model\.ckpt-\d*[123456789]0\..*" | xargs -d"\n" rm

#ls | grep -P "model\.ckpt-\d?\d\d\d\..*" | xargs -d"\n" rm

## Only keep per thousand
ls | grep -P "model\.ckpt-\d*[123456789]\d\d\..*" | xargs -d"\n" rm

#ls | grep -P "model\.ckpt-[12346]?[123456789]\d\d\d\..*" | xargs -d"\n" rm

cd ../..
