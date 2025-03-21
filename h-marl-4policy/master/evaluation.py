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
    total_blue_action_counts = {}
    total_red_action_counts = {}
    total_branch_counts = {"investigate":[], "recover":[], "trafficControl":[]}

    actions_log = []
    obs_log = []
    for i in range(max_eps):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
        count = 0
       
        blue_action_counts = {}
        red_action_counts = {}
        branch_counts = {"0":0, "1":0, "2":0}
        
        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: submission.AGENTS[agent_name].get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name[:12])
                )
                for agent_name in observations
            }
            #print(actions)

            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
            #print(rew)
            done = {
                agent_name: term.get(agent_name, False) or trunc.get(agent_name, False)
                for agent_name in observations
            }
            if all(done.values()):
                break

            r.append(mean(rew.values()))
           
            #print(actions)

            # Store actions for blue agents
            for agent_name in observations:
                #print("here", agent_name)
                #if "master" in agent_name: continue
                branch_counts[str(actions[agent_name])] += 1

                act = cyborg.get_last_action(agent_name[:12])
                act = act[0].name
                if act not in blue_action_counts:
                    blue_action_counts[act] = 0
                blue_action_counts[act] +=  1

            # Store actions for red agents
            for agent_name in red_agents:
                act = cyborg.get_last_action(agent_name[:12])
                act = act[0].name
                if act not in red_action_counts:
                    red_action_counts[act] = 0
                red_action_counts[act] +=  1


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

        for act in blue_action_counts:
            if act not in total_blue_action_counts:
                total_blue_action_counts[act] = []
            total_blue_action_counts[act].append(blue_action_counts[act])

        print("\nStatistics for Blue Team:")
        for act in total_blue_action_counts:
            if len(total_blue_action_counts[act]) < 2:
                print("Episode:", i, act, "mean count:", mean(total_blue_action_counts[act]))
            else:
                print("Episode:", i, act, "mean count:", mean(total_blue_action_counts[act]), "std count:", stdev(total_blue_action_counts[act]))

        for act in red_action_counts:
            if act not in total_red_action_counts:
                total_red_action_counts[act] = []
            total_red_action_counts[act].append(red_action_counts[act])

        print("\nStatistics for Red Team:")
        for act in total_red_action_counts:
            if len(total_red_action_counts[act]) < 2:
                print("Episode:", i, act, "mean count:", mean(total_red_action_counts[act]))
            else:
                print("Episode:", i, act, "mean count:", mean(total_red_action_counts[act]), "std count:", stdev(total_red_action_counts[act]))
        
        total_branch_counts["investigate"].append(branch_counts["0"])
        total_branch_counts["recover"].append(branch_counts["1"])
        total_branch_counts["trafficControl"].append(branch_counts["2"])

        print("\nStatistics per branch (subpolicy):")
        for branch in total_branch_counts:
            if len(total_branch_counts[branch]) < 2:
                print("Episode:", i,  branch, "mean count", mean(total_branch_counts[branch]))
            else:
                print("Episode:", i,  branch, "mean count", mean(total_branch_counts[branch]), stdev(total_branch_counts[branch]))

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
