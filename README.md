# AAMAS-HMARL-submission


This repository contains implementations for Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense paper submitted to AAMAS 2025. We will document pre-requisites and requirements for execution, how to run the code and describe the approach.


## Requirements
Install the 'CybORG' environment provided in [this repository](https://github.com/cage-challenge/cage-challenge-4). This can be done by cloning the repository and then running the following
```
conda create -n cyborg python=3.9
conda activate cyborg
pip install -r Requirements.txt
pip install -e .
```

The environment installation can be tested by running the test provided in the repository itself. Navigate to the directory and run
```
pytest ./CybORG/Tests/test_cc4
```

We use ``ray==2.10.0`` isntead of the latest version. Run the following to override the installation provided by `CybORG` instead
```
pip install ray==2.10.0
```

## Training subpolicies

To train the subpolicies as defined in the paper run the following:
```
cd ./3policy/subpolicies
python3 -u train_subpolicies.py
```

The models will be saved at `models/train_subpolicies`. Copy this and paste it in `3policy/master` and rename the folder `train_subpolicies` to `subpolicies_default`.

## Training master policy
Once the subpolicies have been trained and moved to the `master` directory, we can train the master policy. Run the following:
```
cd ./3policy/master
python3 -u train_master.py
```
## Evaluation and metrics
To see the final score and the metrics defined run the following:
```
cd ./3policy/master
python3 -u evaluation_metrics.py submission.py output
```
This will use the `submission.py` already defined and save the results to a directory `output`

### Above defined steps are similar for `4policy` version as well and can be run by just replacing `3policy` with `4policy`
