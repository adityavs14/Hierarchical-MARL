# Hierarchical MARL


This repository contains the source code and models for the paper "Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense" submitted to AAMAS 2025.


## Requirements
Install the `CybORG` environment provided in [this repository](https://github.com/cage-challenge/cage-challenge-4). Create a virtual environment using `venv` or `conda`. Once the repository is cloned run the following command:
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

## MARL

* Each Blue agent on the Blue Team is trained as an independent learner using a single-policy PPO algorithm. 
* However, the agents are guided by the total team reward and can exchange messages, leading to a multi-agent RL setting. 
* The code and saved models are in the `marl-1policy` directory.
* For training and evaluation, use the same commands discussed below for Hierarchical MARL.

## Hierarchical MARL

* Each Blue agent on the Blue Team is trained as an independent learner, but using a hierarchy of PPO policies. 
* We implemented 2 hierarchical versions based on the number of subpolicies defined:
  * `h-marl-3policy` corresponds to the version with one master policy and 2 subpolicies, namely `Investigate` and `Recover`. 
  * `h-marl-4policy` includes an additional subpolicy `Control Traffic`. 
* The models trained and evaluated in the paper are provided in the `saved_policies` directory.

### H-MARL Expert

This defense strategy corresponds to a master policy guided by expert domain knowledge. The expert knowledge consists in telling the master what subpolicy to call, based on the following rule: "If indicators of compromise are present in the local observation, choose the Recover subpolicy; otherwise, choose the Investigate subpolicy."

For training H-MARL Expert we will only need the `subpolicies` folder. 

#### Training

Navigate to the directory of either `h-marl-3policy` or `h-marl-4policy` depending on the number of subpolicies and run the following commands. We can specify the number of threads to use for parallelization of rollout as an argument. 
```
cd h-marl-3policy
python3 -u subpolicies/train_subpolicies.py 30
```
The above command specifies 30 threads to be used. The models will be saved at `saved_subpolicies/sub`.

#### Evaluation and Metrics
To evaluate H-MARL Expert, run the following command from the h-marl-3policy (or h-marl-4policy) directory.

```
python3 -u subpolicies/evaluation.py subpolicies/submission.py hmarl_expert_output
```
This command will use the `submission.py` already defined in the `subpolicies` directory and save the results to a directory `hmarl_expert_output`.

To collect additional metrics related to network security posture, precision and error of recovery actions, and operational impact, enable the `COMPUTE_METRICS` flag from the `subpolicies/BlueFlatWrapper_CC4.py` script, and run the following command:

```
python3 -u subpolicies/evaluation_metrics.py subpolicies/submission.py hmarl_expert_output
```

### H-MARL Meta
This method corresponds to the 2 step training of subpolicies and master policy. We reuse the subpolicies trained by `H-MARL Expert` to train the Meta master policy. 
#### Training
Ensure that the subpolicies have been trained using the `H-MARL Expert` method defined [here](#h-marl-expert), and run while being in the same directory (e.g. `h-marl-3policy`). We can again pass the number of threads to be used as an argument:
```
python3 -u master/train_master.py 30
```
#### Evaluation and Metrics
To evaluate `H-MARL Meta`, run the following command:
```
python3 -u master/evaluation.py master/submission.py hmarl_meta_output
```
This command will use the `submission.py` already defined in the `master` directory and save the results to a directory `hmarl_meta_output`.

To collect additional metrics related to network security posture, precision and error of recovery actions, and operational impact, enable the `COMPUTE_METRICS` flag from the `master/BlueFlatWrapper_CC4.py` script, and run the following command:

```
python3 -u master/evaluation_metrics.py master/submission.py hmarl_meta_output
```

<!-- 
## Training subpolicies

Navigate to the directory of either `h-marl-3policy` or `h-marl-4policy`, depending on which version of the experiment you want to run. To train the subpolicies as defined in the paper run the following:
```
cd h-marl-3policy
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

This will use the `submission.py` already defined in the `master` directory and save the results to a directory `output`. -->

<!-- ### The same steps described above for `h-marl-3policy` are used to train and evaluate the `h-marl-4policy` version as well.-->

