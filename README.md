# AAMAS-HMARL-submission


This repository contains implementations for Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense paper submitted to AAMAS 2025. We will document pre-requisites and requirements for execution, how to run the code and describe the approach.


## Requirements
Install the 'CybORG' environment provided in [this repository](https://github.com/cage-challenge/cage-challenge-4). Create a virtual environment using `venv` or `conda`. Once the repository is cloned run the following:
```
pip install -r Requirements.txt
pip install -e .
```

The environment installation can be tested by running the test provided in the repository itself. Navigate to the directory and run
```
pytest ./CybORG/Tests/test_cc4
```

Install Ray 2.10:
```
pip install ray==2.10.0
```

## Directories
We have 2 versions based on the number of subpolicies defined. `3policy` corresponds to the version with one master policy and 2 subpolicies, namely `Investigate` and `Recover`. `4policy` includes an additional subpolicy `Control Traffic`. 
Both the settings are currently configured to work on the default settings provided by `CybORG`.

## Training subpolicies

Navigate to the directory of either `3policy` or `4policy`, depending on which version of the experiment you want to run. To train the subpolicies as defined in the paper run the following:
```
cd 3policy
python3 -u subpolicies/train_subpolicies.py
```

The models will be saved at `saved_subpolicies/sub`.

## Training master policy
Once the subpolicies have been trained, we can train the master policy. Run the following:
```
python3 -u master/train_master.py
```
## Evaluation and metrics
To see the final score and the metrics defined run the following:
```
python3 -u master/evaluation_metrics.py master/submission.py output
```
This will use the `submission.py` already defined in the `master` directory and save the results to a directory `output`.

### Above defined steps are similar for `4policy` version as well and can be run by just replacing `3policy` with `4policy`
