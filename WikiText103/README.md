# Experiment for WikiText-103 task


## Requirement

Please refer to [link](https://github.com/IDSIA/lmtool-fwp) to bulid experimental environment.


## Prepare data

``` {bash}
./getdata.sh
```



## Training

``` {bash}
cd src
```
- Linear Transformer with AID network
``` {bash}
bash ../example_scripts//run.sh train --work_dir linear_with_aid --seed 1111 --n_decomposition_layer 4 --attn_type 9998
```
- Delta Network with AID network
``` {bash}
bash ../example_scripts//run.sh train --work_dir linear_with_delta --seed 1111 --n_decomposition_layer 4 --attn_type 9999
```


## Validation

``` {bash}
bash ../example_scripts/run.sh valid --work_dir ${work_dir}
```

## Test

``` {bash}
bash ../example_scripts/run.sh eval --work_dir ${work_dir}
```
