import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from datetime import datetime

import json

import sys
import os
import itertools

def rmkdir(path: str):
    """Recursive mkdir"""
    partial_path = ""
    for p in path.split("/"):
        partial_path += p + "/"

        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            if os.path.isfile(partial_path):
                raise RuntimeError(f"Cannot create {partial_path} (exists as file).")

        os.mkdir(partial_path)


def load_submission(source: str):
    """Load submission from a directory or zip file"""
    sys.path.insert(0, source)

    if source.endswith(".zip"):
        try:
            # Load submission from zip.
            from submission.submission import Submission
        except ImportError as e:
            raise ImportError(
                """
                Error loading submission from zip.
                Please ensure the zip contains the path submission/submission.py
                """
            ).with_traceback(e.__traceback__)
    else:
        # Load submission normally
        from submission import Submission

    # Remove submission from path.
    sys.path.remove(source)
    return Submission


def run_evaluation(submission, log_path, max_eps=100, write_to_file=True, seed=None):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)
    red_agents = [red_agent for red_agent in cyborg.agents if 'red' in red_agent]

    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    start = datetime.now()

    total_reward = []
    actions_log = []
    obs_log = []

    infection_metric = {}
    noninfection_frac = {}
    noninfection_frac_crt_mean = {}
    nonprivileged_frac = {}
    nonprivileged_frac_crt_mean = {}
    red_actions = {}
    red_actions_crt_mean = {}

    true_false_pos = {}
    infection_stride = {}
    privileged_stride = {}

    for i in range(max_eps):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
       
        infection_metric[i] = {}
        noninfection_frac[i] = {}
        nonprivileged_frac[i] = {}
        red_actions[i] = {}

        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: submission.AGENTS[agent_name].get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name[:12])
                )
                for agent_name in observations
            }
            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
            #print(rew)

            r.append(mean(rew.values()))
             
            # compromise metric
            imetric, nonimetric, privileged, nonprivileged = wrapped_cyborg.infection_metric()
            for subnet in imetric:
                if  subnet not in infection_metric[i]:
                    infection_metric[i][subnet] = []
                    noninfection_frac[i][subnet] = []
                    nonprivileged_frac[i][subnet] = []

                infection_metric[i][subnet].append(imetric[subnet])

                frac = nonimetric[subnet] / (nonimetric[subnet] + imetric[subnet])
                noninfection_frac[i][subnet].append(frac)
                
                pfrac = nonprivileged[subnet] / (nonprivileged[subnet] + privileged[subnet])
                nonprivileged_frac[i][subnet].append(pfrac)
                
            #  operations metric
            # Store actions for red agents
            for agent_name in red_agents:
                acti = cyborg.get_last_action(agent_name)
                act = acti[0].name
                if act != "Impact": continue

                # although red can call Impact in other subnets, 
                # it is only effective within the operational subnet
                # where the OT services reside
                if act == "Impact":
                    if "operational" not in acti[0].hostname: continue
                    #print(acti[0].hostname)
                    
                if agent_name not in red_actions[i]:
                    red_actions[i][agent_name] = {}

                if act not in red_actions[i][agent_name]:
                    red_actions[i][agent_name][act] = 0

                red_actions[i][agent_name][act] +=  1

            done = {
                agent_name: term.get(agent_name, False) or trunc.get(agent_name, False)
                for agent_name in observations
            }
            if all(done.values()):

                break

            if write_to_file:
                a.append(actions)
                o.append(
                    {
                        agent_name: observations[agent_name]
                        for agent_name in observations.keys()
                    }
                )
                

        total_reward.append(sum(r))
        print("\nEpisode:", i, mean(total_reward))

        print("Infection:")
        #for subnet in infection_metric[i]:
        #    print(subnet, "noninfection:", mean(noninfection_frac[i][subnet]), "nonprivileged:", mean(nonprivileged_frac[i][subnet]), mean(infection_metric[i][subnet]), stdev(infection_metric[i][subnet]))
       
        noninfection_frac_all = []
        nonprivileged_frac_all = []

        for subnet in noninfection_frac[i]:
            if subnet not in noninfection_frac_crt_mean:
                noninfection_frac_crt_mean[subnet] = []
                nonprivileged_frac_crt_mean[subnet] = []
                
            noninfection_frac_crt_mean[subnet].append(mean(noninfection_frac[i][subnet]))
            nonprivileged_frac_crt_mean[subnet].append(mean(nonprivileged_frac[i][subnet]))
            print("running avg", subnet, "noninfection:", round(mean(noninfection_frac_crt_mean[subnet]), 2), "nonprivileged:", round(mean(nonprivileged_frac_crt_mean[subnet]), 2))
       
            if "contractor" not in subnet:
                noninfection_frac_all.extend(noninfection_frac_crt_mean[subnet])
                nonprivileged_frac_all.extend(nonprivileged_frac_crt_mean[subnet])

        #noninfection_frac_all = list(itertools.chain.from_iterable(noninfection_frac_crt_mean.values()))
        print("Noninfection frac across subnets:", round(mean(noninfection_frac_all), 2))
        #nonprivileged_frac_all = list(itertools.chain.from_iterable(nonprivileged_frac_crt_mean.values()))
        print("Nonprivileged frac across subnets:", round(mean(nonprivileged_frac_all), 2))

        print("\nTrue False pos:")
        tp_fp = wrapped_cyborg.true_false_pos
        for subnet in tp_fp:
            if subnet not in true_false_pos:
                true_false_pos[subnet] = {}
                true_false_pos[subnet]["tp"] = []
                true_false_pos[subnet]["fp"] = []

            true_false_pos[subnet]["tp"].append(tp_fp[subnet]["tp"])
            true_false_pos[subnet]["fp"].append(tp_fp[subnet]["fp"])

        true_pos_all = []
        false_pos_all = []
        for subnet in true_false_pos:  # the running average
            print(subnet, "tp", round(mean(true_false_pos[subnet]["tp"]), 2), "fp", round(mean(true_false_pos[subnet]["fp"]), 2))
            true_pos_all.append(round(mean(true_false_pos[subnet]["tp"]), 2))
            false_pos_all.append(round(mean(true_false_pos[subnet]["fp"]), 2))
        print("True pos across subnets:", round(mean(true_pos_all), 2))
        print("False pos across subnets:", round(mean(false_pos_all), 2))

        print("\nInfection stride length:")
        stride_crt = wrapped_cyborg.infection_stride_lengths
        for subnet in stride_crt:
            if len(stride_crt[subnet]) == 0: continue
            if subnet not in infection_stride:
                infection_stride[subnet] = []
            infection_stride[subnet].append(mean(stride_crt[subnet])) # add the average stride for this episode

        for subnet in infection_stride:
            print(subnet, "stride", round(mean(infection_stride[subnet]), 2)) # the running average up to the current episode
        
        infection_stride_all = list(itertools.chain.from_iterable(infection_stride.values()))
        print("Mean stride across subnets:", round(mean(infection_stride_all), 2))

        print("\nPrivileged stride length:")
        priv_stride_crt = wrapped_cyborg.privileged_stride_lengths
        for subnet in priv_stride_crt:
            if len(priv_stride_crt[subnet]) == 0: continue 
            if subnet not in privileged_stride:
                privileged_stride[subnet] = []
            privileged_stride[subnet].append(mean(priv_stride_crt[subnet])) # add the average stride for this episode

        for subnet in privileged_stride:
            print(subnet, "stride", round(mean(privileged_stride[subnet]),2)) # the running average up to the current episode
        
        privileged_stride_all = list(itertools.chain.from_iterable(privileged_stride.values()))
        print("Mean stride  across subnets:", round(mean(privileged_stride_all), 2))

        print("\nImpact Actions:")
        for agent_name in red_agents:
            #print("\n", agent_name, wrapped_cyborg.subnet_assignment(agent_name))
            if agent_name not in red_actions_crt_mean:
                red_actions_crt_mean[agent_name] = {}
            if agent_name not in red_actions[i]: 
                act = "Impact"
                if act  not in red_actions_crt_mean[agent_name]:
                    red_actions_crt_mean[agent_name][act] = []
                red_actions_crt_mean[agent_name][act].append(0)
                #print(agent_name, wrapped_cyborg.subnet_assignment(agent_name), act, mean(red_actions_crt_mean[agent_name][act]))
                print(agent_name, wrapped_cyborg.subnet_assignment(agent_name), act, red_actions_crt_mean[agent_name][act], "mean", round(mean(red_actions_crt_mean[agent_name][act]), 2))
                continue

            for act in red_actions[i][agent_name]:
                if act not in red_actions_crt_mean[agent_name]:
                    red_actions_crt_mean[agent_name][act] = []
                red_actions_crt_mean[agent_name][act].append(red_actions[i][agent_name][act])
            for act in red_actions_crt_mean[agent_name]:
                if act == 'Sleep': continue
                print(agent_name, wrapped_cyborg.subnet_assignment(agent_name), act, red_actions_crt_mean[agent_name][act], "mean", round(mean(red_actions_crt_mean[agent_name][act]), 2))
            
        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward)
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                },
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
            }
            json.dump(data, output)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")


if __name__ == "__main__":
    import argparse
    from train_master import algo_builder as master_algo
    sys.path.append('..')
    from subpolicies.train_subpolicies import algo_builder as sub_algo
    sys.path.remove("..")

    parser = argparse.ArgumentParser("CybORG Evaluation Script")
    parser.add_argument("submission_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "--append-timestamp",
        action="store_true",
        help="Appends timestamp to output_path",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set the seed for CybORG"
    )
    parser.add_argument("--max-eps", type=int, default=100, help="Max episodes to run")
    args = parser.parse_args()
    args.output_path = os.path.abspath(args.output_path)
    args.submission_path = os.path.abspath(args.submission_path)

    if not args.output_path.endswith("/"):
        args.output_path += "/"

    if args.append_timestamp:
        args.output_path += time.strftime("%Y%m%d_%H%M%S") + "/"

    rmkdir(args.output_path)
    
    # if the policies fail to load sometimes loading and saving locally helps. 
    # This can happen due to different ray versions as well. 
    # Uncomment the block below to reconfigure the saved policies
    
    # algo = sub_algo()
    # checkpoint_path = "./saved_policies/sub/iter_49"
    # algo.restore(checkpoint_path)
    # algo.save(checkpoint_path)
    
    # algo = master_algo()
    # checkpoint_path = "./saved_policies/master/iter_49"
    # algo.restore(checkpoint_path)
    # algo.save(checkpoint_path)

    submission = load_submission(args.submission_path)
    run_evaluation(
        submission, max_eps=args.max_eps, log_path=args.output_path, seed=args.seed
    )
