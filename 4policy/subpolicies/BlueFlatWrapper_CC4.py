from __future__ import annotations

from gymnasium import Space, spaces

from CybORG import CybORG
from CybORG.Simulator import State
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator,
)

from typing import Any
import pprint
import numpy as np
import networkx as nx

import functools
import itertools
import random
from operator import ior
from functools import reduce

# no changes to the Action Wrapper
# messages are created in this wrapper
# this wrapper also handles masking
# and the observation space change
from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (
    BlueFixedActionWrapper,
    MESSAGE_LENGTH,
    EMPTY_MESSAGE,
    NUM_MESSAGES,
    SUBNET_USER_FORMAT,
    SUBNET_SERVER_FORMAT,
    SUBNET_ROUTER_FORMAT
)

from Ray_BlueAgent import HMARLBlueAgent

NUM_SUBNETS = 9
NUM_HQ_SUBNETS = 3

MAX_USER_HOSTS = EnterpriseScenarioGenerator.MAX_USER_HOSTS
MAX_SERVER_HOSTS = EnterpriseScenarioGenerator.MAX_SERVER_HOSTS
MAX_HOSTS = MAX_USER_HOSTS + MAX_SERVER_HOSTS

subnets_list_blue = ['public_access_zone_subnet', 'operational_zone_a_subnet', 'operational_zone_b_subnet', 'restricted_zone_a_subnet', 'restricted_zone_b_subnet', 'admin_network_subnet', 'office_network_subnet']
NUM_RED_AGENTS = 6

USE_BLOCK = True
USE_FILES = True
USE_DECOYS= True
USE_MESSAGES = True

class BlueFlatWrapper(BlueFixedActionWrapper):
    """Converts observation spaces to vectors of fixed size and ordering across episodes.

    This is a companion wrapper to the BlueFixedActionWrapper and inherits the fixed
    action space and int-to-action mappings as a result.

    Using the *sorted* host and subnet lists from FixedAction wrapper, this wrapper
    establishes the maximum observation space for each agent. On each step, the
    observation vectors are populated such that each element within a vector will
    have a consistent meaning across runs. This is critical for RL-based agents.
    """

    def __init__(self, env: CybORG, *args, **kwargs):
        """Initialize the BlueFlatWrapper for blue agents.

        Note: The padding setting is inherited from BlueFixedActionWrapper.

        Args:
            env (CybORG): The environment to wrap.

            *args, **kwargs: Extra arguments are ignored.
        """
        super().__init__(env, *args, **kwargs)

        # hosts in this scenario
        self.hostnames = list(env.environment_controller.state.hosts.keys())

        self._short_obs_space, self._long_obs_space = self._get_init_obs_spaces()
        self.comms_policies = self._build_comms_policy()
        self.policy = {}
        self.initialize_action_indexes()

        # initialize history for malicious files
        if USE_FILES == True:
            self.files = {}
            for a in self.agents:
                self.files[a] = {}
                for subnet in self.subnets(a):
                    self.files[a][subnet] = [0] * MAX_HOSTS

        # initialize history for suspicious host events, i.e., processes and connections
        self.suspicious_processes = {}
        self.suspicious_connections  = {}

        for a in self.agents:
            self.suspicious_processes[a] = {}
            self.suspicious_connections[a] = {}

            for subnet in self.subnets(a):
                self.suspicious_processes[a][subnet] = [False] * MAX_HOSTS
                self.suspicious_connections[a][subnet] = [False] * MAX_HOSTS

        # initialize history of hosts where decoy access originated from
        if USE_DECOYS == True:
            self.decoy_access_from = {}
            self.decoy_msg = {}
            for a in self.agents:
                self.decoy_access_from[a] = {}
                self.decoy_msg[a] = set()
                for subnet in self.subnets(a):
                    self.decoy_access_from[a][subnet] = [False] * MAX_HOSTS

        # will use the subnet info for decoys/messages
        subnet_names = sorted(list(self.env.environment_controller.state.subnet_name_to_cidr.keys()))
        self.subnet_names = [name.lower() for name in subnet_names]

        # env variables
        self.output = {}
        
        self.investigate_mask = {}
        self.recover_mask = {}
        self.trafficControl_mask = {}
        self.high_actionmask = {}

        self.prev_rewards = {}
        self.prev_term = {}
        self.prev_trunc = {}
        self.return_agent = {}
        
        self.obs_recover = {} # will constitute the reduced observation space 
        self.obs_investigate = {} # will constitute the reduced observation space
        self.obs_trafficControl = {}

        self.inv_policies = {}
        self.rec_policies = {}

        for name in self.agents:
            self.inv_policies[name] = HMARLBlueAgent(name= f"Agent{name[-1]}_investigate")
            self.rec_policies[name] = HMARLBlueAgent(name= f"Agent{name[-1]}_recover")


    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and update the observation space.

        Args: All arguments are forwarded to the env provided to __init__.

        Returns
        -------
        observation : dict[str, Any]
            The observations corresponding to each agent, translated into a vector format.
        info : dict[str, dict]
            Forwarded from self.env.
        """
        observations, info = super().reset(*args, **kwargs)
        self.comms_policies = self._build_comms_policy()
        self.initialize_action_indexes()
        
        # will use the subnet info for decoys/messages
        subnet_names = sorted(list(self.env.environment_controller.state.subnet_name_to_cidr.keys()))
        self.subnet_names = [name.lower() for name in subnet_names]

        # initialize history for malicious files
        if USE_FILES == True:
            self.files = {}
            for a in self.agents:
                self.files[a] = {}
                for subnet in self.subnets(a):
                    self.files[a][subnet] = [0] * MAX_HOSTS

        # initialize history for suspicious host events, i.e., processes and connections
        self.suspicious_processes = {}
        self.suspicious_connections  = {}

        for a in self.agents:
            self.suspicious_processes[a] = {}
            self.suspicious_connections[a] = {}

            for subnet in self.subnets(a):
                self.suspicious_processes[a][subnet] = [False] * MAX_HOSTS
                self.suspicious_connections[a][subnet] = [False] * MAX_HOSTS

        # each agent marks hosts (from its own subnet) that accessed decoys
        if USE_DECOYS == True:
            self.decoy_access_from = {}
            self.decoy_msg = {}
            for a in self.agents:
                self.decoy_access_from[a] = {}
                self.decoy_msg[a] = set()
                for subnet in self.subnets(a):
                    self.decoy_access_from[a][subnet] = [False] * MAX_HOSTS

        self.output = {}
        self.investigate_mask = {}
        self.recover_mask = {}
        self.trafficControl_mask = {}
        
        self.prev_rewards = {f"blue_agent_{i}": 0 for i in range(5)}
        self.prev_term = {f"blue_agent_{i}": False for i in range(5)}
        self.prev_trunc = {f"blue_agent_{i}": False for i in range(5)}

        ### defining subpolicy agents ###
        obs = {}
        inf = {}

        for a in self.agents:
            [self.output[a], 
             self.obs_investigate[a], 
             self.obs_recover[a], 
             self.obs_trafficControl[a],
             self.investigate_mask[a], 
             self.recover_mask[a], 
             self.trafficControl_mask[a]
            ] = self.observation_change(a, observations[a])

            # self.high_actionmask is [0,1] if master chose recover in observation_change; [1,0] if master chose investigate
            investigate = a+"_investigate"
            obs[investigate] = {'observations': self.obs_investigate[a], 'action_mask': self.investigate_mask[a]}
            inf[investigate] = {'action_mask': self.investigate_mask[a]}
            self.high_actionmask[a] = np.array([1,0,0], dtype=np.int64)
            self.return_agent[a] = investigate

        return obs, inf


    def step(
        self,
        actions: dict[str, int | Action] = None,
        messages: dict[str, Any] = None,
        **kwargs,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Take a step in the enviroment.

        Parameters:
            action_dict : dict[str, int | Action]
                The action or action index corresponding to each agent. 
                Indices will be mapped to CybORG actions using the equivalent of `actions(k)[v]`. 
                The meaning of each action can be found using `action_labels(k)[v]`.
            messages : dict[str, Any]
                Messages from each agent. If an agent does not specify a message, it will send an empty message.
            **kwargs : dict[str, Any]
                Extra keywords are forwarded.

        Returns
        -------
        observation : dict[str, np.ndarray] 
            Observations for each agent as vectors.
        rewards : dict[str, float] 
            Rewards for each agent.
        terminated : dict[str, bool]
            Flags whether the agent finished normally.
        truncated : dict[str, bool]
            Flags whether the agent was stopped by env.
        info : dict[str, dict]
            Forwarded from BlueFixedActionWrapper.
        """

        for pol in actions:
            if "master" in pol:
                break
            else:
                #print(actions)
                for agent in actions:
                    # just re-use investigate and recover policies, and only learn trafficControl
                    #print("before", agent, actions[agent])
                    if "investigate" in agent:
                        actions[agent] = self.inv_policies[agent[:12]].get_action(
                            { 'observations':self.obs_investigate[agent[:12]],
                              'action_mask':self.investigate_mask[agent[:12]]
                            },
                            self.action_space(agent[:12])
                        )
                    if "recover" in agent:
                        actions[agent] = self.rec_policies[agent[:12]].get_action(
                        { 'observations':self.obs_recover[agent[:12]],
                          'action_mask':self.recover_mask[agent[:12]]
                        },
                        self.action_space(agent[:12])
                        )
                    #print("after",  agent, actions[agent])

                return self.env_step(actions)

        obs = {}
        inf = {}
        rew = {}
        term = {}
        trunc = {}

        #print("wrapper master step actions:", actions)
        for a in actions:
            rew[a] = self.prev_rewards[a[:12]]
            term[a] = self.prev_term[a[:12]]
            trunc[a] = self.prev_trunc[a[:12]]
            if actions[a] == 0:
                policy = a[:12]+"_investigate"
                obs[policy] = {"observations": self.obs_investigate[a[:12]], "action_mask":self.investigate_mask[a[:12]]}
                inf[policy] = {'action_mask': self.investigate_mask[a[:12]]}
                self.high_actionmask[a] = np.array([1,0,0], dtype=np.int64)

            elif actions[a] == 1:
                policy = a[:12]+"_recover"
                obs[policy] = {"observations": self.obs_recover[a[:12]], "action_mask":self.recover_mask[a[:12]]}
                inf[policy] = {'action_mask': self.recover_mask[a[:12]]}
                self.high_actionmask[a] = np.array([0,1,0], dtype=np.int64)
            
            elif actions[a] == 2:
                policy = a[:12]+"_trafficControl"
                obs[policy] = {"observations": self.obs_trafficControl[a[:12]], "action_mask":self.trafficControl_mask[a[:12]]}
                inf[policy] = {'action_mask': self.trafficControl_mask[a[:12]]}
                self.high_actionmask[a] = np.array([0,0,1], dtype=np.int64)

            if self.return_agent[a[:12]] != policy:
                print("\nmaster did not choose right!", self.return_agent[a[:12]], policy)
                exit()

        #print("end of step", obs, rew, term, trunc, inf)
        return obs, rew, term, trunc, inf


    def env_step(self, actions, messages = None, **kwargs):
        temp_actions = {}
        for agent in actions:
            temp_actions[agent[:12]] = actions[agent]

        messages = {} if messages is None else messages

        # update messages; agent is sending
        # message encoding relies on decoy access info
        if USE_MESSAGES == True:
            for agent in list(temp_actions.keys()):  
                # messages[agent] = self._encode_messages(agent)
                messages[agent] = self._encode_messages_v2(agent) # mark in which subnet you have IOCs

        #print("subpolicy stepping in env", temp_actions)
        observations, rewards, terminated, truncated, info = super().step(
            actions=temp_actions, messages=messages, **kwargs
        )

        obs = {}
        rew = {}
        term = {}
        trunc = {}
        inf = {}
        for agent, o in sorted(observations.items()):
            if "blue" in agent:
                [self.output[agent], 
                 self.obs_investigate[agent], 
                 self.obs_recover[agent], 
                 self.obs_trafficControl[agent],
                 self.investigate_mask[agent], 
                 self.recover_mask[agent], 
                 self.trafficControl_mask[agent]] = self.observation_change(agent, o)

                # back to master
                master = agent+"_master"
                obs[master] = {"observations": self.output[agent], 'action_mask': self.high_actionmask[agent], 'id':1}
                rew[[x for x in actions.keys() if agent in x][0]] = rewards[agent]

                self.prev_rewards[agent] = rewards[agent]
                self.prev_term[agent] = terminated[agent]
                self.prev_trunc[agent] = truncated[agent]

                term[[x for x in actions.keys() if agent in x][0]] = terminated[agent]
                trunc[[x for x in actions.keys() if agent in x][0]] = truncated[agent]
                inf[master] = {'action_mask': self.high_actionmask[agent]}

        return obs, rew, term, trunc, inf


    def _get_init_obs_spaces(self):
        """Calculates the size of the largest observation space for each agent."""

        if USE_FILES == False and USE_DECOYS == False:
            host_iocs = []
        else:
            iocs_num = 1

            if USE_FILES == True:
                iocs_num += 2
 
            if USE_DECOYS == True:
                iocs_num += 1

            host_iocs = MAX_HOSTS * [iocs_num]
        
        observation_space_components = {
            "mission": [3],
            "subnet": NUM_SUBNETS * [2],
            "blocked_subnets": NUM_SUBNETS * [2],
            "comms_policy": NUM_SUBNETS * [2],
            "malicious_processes": MAX_HOSTS * [2],
            "network_connections": MAX_HOSTS * [2],
            "host_iocs": host_iocs,
            "network_iocs": NUM_SUBNETS * [2],
        }

        observation_head = observation_space_components["mission"]
        
        # messages received through the observation dictionary 
        # are processed and their info is added to the host_iocs subvector
        # observation_tail = observation_space_components["messages"]
        observation_tail = []

        observation_middle = list(
            itertools.chain(
                *[
                    v
                    for k, v in observation_space_components.items()
                    if k not in ("mission", "messages")
                ]
            )
        )

        short_observation_components = (
            observation_head + observation_middle + observation_tail
        )

        long_observation_components = (
            observation_head + (NUM_HQ_SUBNETS * observation_middle) + observation_tail
        )

        short_observation_space = spaces.MultiDiscrete(short_observation_components)
        long_observation_space = spaces.MultiDiscrete(long_observation_components)

        self._observation_space = {
            agent: long_observation_space
            if self.is_padded or agent == "blue_agent_4"
            else short_observation_space
            for agent in self.agents
        }

        # using masking in Ray
        # just defining the spaces
        observation_space_recover = {}
        observation_space_investigate = {}
        observation_space_trafficControl = {}
                
        # just [mission,] processes, connections, iocs
        investigate_components = list(
            itertools.chain(
                *[
                    v
                    for k, v in observation_space_components.items()
                    if k in ("malicious_processes", "network_connections", "host_iocs")
                ]
            )   
        )

        # just [mission and] iocs
        recover_components = host_iocs

        trafficControl_components = list(
            itertools.chain(
                *[
                    v
                    for k, v in observation_space_components.items()
                    if k in ("subnet", "blocked_subnets", "comms_policy", "network_iocs")
                ]
            )   
        )

        for agent in self.agents:
            if agent == "blue_agent_4":
                observation_space_recover[agent] = spaces.MultiDiscrete(observation_head + NUM_HQ_SUBNETS * recover_components)
                observation_space_investigate[agent] = spaces.MultiDiscrete(observation_head + NUM_HQ_SUBNETS * investigate_components)
                observation_space_trafficControl[agent] = spaces.MultiDiscrete(observation_head + NUM_HQ_SUBNETS * trafficControl_components)
            else:
                observation_space_recover[agent] = spaces.MultiDiscrete(observation_head + recover_components)
                observation_space_investigate[agent] = spaces.MultiDiscrete(observation_head + investigate_components)
                observation_space_trafficControl[agent] = spaces.MultiDiscrete(observation_head + trafficControl_components)

        self._observation_space = {
                agent: spaces.Dict({"observations": self._observation_space[agent],
                                    "obs_recover": observation_space_recover[agent],
                                    "obs_investigate": observation_space_investigate[agent],
                                    "obs_trafficControl": observation_space_trafficControl[agent],
                        "action_mask": spaces.MultiDiscrete([2] * len(self.action_mask(agent))),
                        })
            for agent in self.agents
        }

        return short_observation_space, long_observation_space


    def initialize_action_indexes(self):
        self.monitor_indexes = {
            a: self.get_action_indexes(a, "Monitor") for a in self.agents
        }

        self.sleep_indexes = {
            a: self.get_action_indexes(a, "Sleep") for a in self.agents
        }

        self.restore_indexes = {
            a: self.get_subnet_action_indexes(a, "Restore") for a in self.agents
        }

        self.remove_indexes = {
            a: self.get_subnet_action_indexes(a, "Remove") for a in self.agents
        }
         
        self.block_traffic_indexes = {
            a: self.get_subnet_action_indexes(a, "BlockTraffic") for a in self.agents
        }

        self.allow_traffic_indexes = {
            a: self.get_subnet_action_indexes(a, "AllowTraffic") for a in self.agents
        }
        
        #self.decoys = {
        #    a: self.get_action_indexes(a, "Decoy") for a in self.agents
        #}

        self.decoy_indexes = {
            #a: self.get_subnet_action_indexes(a, "Decoy") for a in self.agents
            a: self.get_action_indexes(a, "Decoy") for a in self.agents
        }

        self.analyse_indexes = {
            a: self.get_action_indexes(a, "Analyse") for a in self.agents
            # a: self.get_subnet_action_indexes(a, "Analyse") for a in self.agents
        }

    
    def get_action_indexes(self, agent_name: str, action_name: str):
        actions = self.get_action_space(agent_name)['actions']
        action_indexes = []

        for i in range(len(actions)):
            if action_name in actions[i].name:
                action_indexes.append(i)

        return action_indexes

    
    def get_subnet_action_indexes(self, agent_name, action_name):
        actions = self.get_action_space(agent_name)['actions']
        labels = self.get_action_space(agent_name)['labels']
        action_indexes = {}

        for i in range(len(actions)):
            if action_name not in labels[i]: continue
            words = labels[i].split()
            
            if 'Invalid' not in words[0]:
                assert (len(words) > 1 and "subnet" in words[1])
                subnet_str = words[1]
            else:
                assert (len(words) > 2 and "subnet" in words[2])
                subnet_str = words[2]

            for subnet in subnets_list_blue:
                if subnet_str.startswith(subnet): break
             
            if subnet not in action_indexes:
                action_indexes[subnet] = []
            action_indexes[subnet].append(i)

        return action_indexes


    def get_mission_phase(self):
        return self.env.environment_controller.state.mission_phase


    def observation_change(self, agent_name: str, observation: dict) -> np.ndarray:
        """Converts an observation dictionary to a vector of fixed size and ordering.

        Parameters
        ----------
        agent_name : str
            Agent corresponding to the observation.
        observation : dict 
            Observation to convert to a fixed vector.

        Returns
        -------
        output : np.ndarray

        """

        if USE_MESSAGES == True:
            # Messages from other agents
            # This assumes CybORG provides a consistent ordering.
            messages = observation.get("message", [EMPTY_MESSAGE] * NUM_MESSAGES)
            assert len(messages) == NUM_MESSAGES

            # update the self.decoy_access_from with info from other agents
            # find out if other agents captured decoy attempts originating in this agent's subnets
            #self._decode_messages(agent_name, messages)

            # messages tell in which subnets are IOCs
            network_iocs_subvector = self._decode_messages_v2(agent_name, messages)            
            #print(agent_name, "network_iocs_subvector", network_iocs_subvector)


        state = self.env.environment_controller.state

        #pprint.pprint(observation)

        proto_observation = []
        proto_recover_subvector = []
        proto_investigate_subvector = []
        proto_trafficControl_subvector = []

        # will need to update the mask
        new_mask = self.action_mask(agent_name)

        # Mission Phase
        mission_phase = state.mission_phase
        proto_observation.append(mission_phase)
        proto_recover_subvector.append(mission_phase)
        proto_investigate_subvector.append(mission_phase)
        proto_trafficControl_subvector.append(mission_phase)

        # Useful (sorted) information
        sorted_subnet_name_to_cidr = sorted(state.subnet_name_to_cidr.items())

        subnet_names, subnet_cidrs = zip(*sorted_subnet_name_to_cidr)
        subnet_names = [name.lower() for name in subnet_names]
        hosts = self.hosts(agent_name)
       
        iocs_exist = False

        if USE_FILES == True:
             h_analyse, files = self.get_files_from_analyse(observation)

        for subnet in self.subnets(agent_name):
            # One-hot encoded subnet vector
            subnet_subvector = [subnet == name for name in subnet_names]

            # Get blocklist
            blocked_subnets = state.blocks.get(subnet, [])
            blocked_subvector = [s in blocked_subnets for s in subnet_names]
           
            # Comms
            comms_policy = self.comms_policies[state.mission_phase]
            comms_matrix = nx.to_numpy_array(comms_policy, nodelist=subnet_names)
            comms_policy_subvector = comms_matrix[subnet_names.index(subnet)]
            comms_policy_subvector = np.logical_not(comms_policy_subvector)
            self.policy[agent_name] = comms_policy

            # Process malware events for users, then servers
            subnet_hosts = [h for h in hosts if subnet in h and "router" not in h]
           
            # erase history for remove/restore actions, if necessary
            self.erase_history_if_recovered(agent_name, observation, subnet, subnet_hosts)

            process_subvector = [
                h in state.hosts and 0 < len(self._get_procesess(state, h))
                for h in subnet_hosts
            ]

            connection_subvector = [
                h in state.hosts and 0 < len(self._get_connections(state, h))
                for h in subnet_hosts
            ]
         
            # update subvectors with the history
            for i in range(len(subnet_hosts)):
                process_subvector[i] = process_subvector[i] or self.suspicious_processes[agent_name][subnet][i]
                connection_subvector[i] = connection_subvector[i] or self.suspicious_connections[agent_name][subnet][i]

                self.suspicious_processes[agent_name][subnet][i] = process_subvector[i]
                self.suspicious_connections[agent_name][subnet][i] = connection_subvector[i]
               
            host_iocs_subvector = MAX_HOSTS * [0]

            # less priority (3), scanning stage
            if USE_DECOYS == True:
                for i in range(len(subnet_hosts)):
                    h = subnet_hosts[i]
                    if h not in state.hosts: continue

                    # find out where accesses to decoys on h came from in this timestep
                    # update this info in self.decoy_access_from[a][subnet]
                    self.update_who_accessed_decoys(agent_name, h)

                    if self.decoy_access_from[agent_name][subnet][i] == True:
                        host_iocs_subvector[i] = 3
                        iocs_exist = True
                        break


            if USE_FILES == True:
                # if analyse has been executed successfully, update files history
                if  h_analyse != None and len(files) > 0 and h_analyse in subnet_hosts:
                    h_idx = subnet_hosts.index(h_analyse)
                    self.files[agent_name][subnet][h_idx] = 2 # user access, priority 2, exploit stage

                    # overwrite with root access if present
                    for f in files:
                        if 'escalate.sh' in f['File Name']:
                            self.files[agent_name][subnet][h_idx] = 1 # root access, priority 1, escalation stage
                            break
                    
                for i in range(len(subnet_hosts)):
                    h = subnet_hosts[i]
                    if h not in state.hosts: continue
                    if self.files[agent_name][subnet][i] != 0:

                        # full subpolicy learning version
                        host_iocs_subvector[i] = self.files[agent_name][subnet][i]
                        iocs_exist = True

            #print(agent_name, subnet, "host_iocs_subvector", host_iocs_subvector)    
            
            proto_observation.extend(
                itertools.chain(
                    subnet_subvector,
                    blocked_subvector,
                    comms_policy_subvector,
                    process_subvector,
                    connection_subvector,
                    host_iocs_subvector,
                    network_iocs_subvector,
                )
            )

            proto_recover_subvector.extend(host_iocs_subvector) # used for reduced obs_space
            
            proto_investigate_subvector.extend(
                itertools.chain(
                    process_subvector,
                    connection_subvector,
                    host_iocs_subvector,
                )
            )

            proto_trafficControl_subvector.extend(
                itertools.chain(
                    subnet_subvector,
                    blocked_subvector,
                    comms_policy_subvector,
                    network_iocs_subvector,
                )
            )

            # if the mission's policy does not allow traffic control between subnets, mask out those actions
            action_indexes_allow = self.allow_traffic_indexes[agent_name][subnet]
            action_indexes_block = self.block_traffic_indexes[agent_name][subnet]
           
            j = 0
            for i in range(len(subnet_names)):
                if i == subnet_names.index(subnet): # this is same subnet, can't block/allow connections to myself
                    continue
                    
                # value of 1 in comms_policy_subvector means blocked traffic, so mask should be set to False 
                index_allow = action_indexes_allow[j]
                index_block = action_indexes_block[j]

                if USE_BLOCK == True:
                    new_mask[index_allow] = not bool(comms_policy_subvector[i])
                    new_mask[index_block] = not bool(comms_policy_subvector[i])
                else:
                    new_mask[index_allow] = False
                    new_mask[index_block] = False

                j = j + 1
                
        
        output = np.array(proto_observation, dtype=np.int64)
        output_recover = np.array(proto_recover_subvector, dtype=np.int64)
        output_investigate = np.array(proto_investigate_subvector, dtype=np.int64)
        output_trafficControl = np.array(proto_trafficControl_subvector, dtype=np.int64)

        # Apply padding as required
        if self.is_padded:
            output = np.pad(
                output, (0, self._long_obs_space.shape[0] - output.shape[0])
            )

        # handling extra ticks
        # if the action is in progress, the environment executes sleep
        # we make sure only sleep is available
        # monitor runs at every step, so we still needed to update the histories earlier in the code

        if observation['success'].name == 'IN_PROGRESS':
            amsize = len(self.action_mask(agent_name))
            new_mask_investigate = [False] * amsize
            new_mask_recover = [False] * amsize
            new_mask_trafficControl = [False] * amsize

            # action index of sleep
            my_labels = self.action_labels(agent_name)
            index_sleep = my_labels.index("Sleep")

            new_mask_investigate[index_sleep] = True
            new_mask_recover[index_sleep] = True
            new_mask_trafficControl[index_sleep] = True

            output = np.ones(len(self._observation_space[agent_name]["observations"]), dtype=np.int64)
            output_recover = np.ones(len(output_recover), dtype=np.int64)
            output_investigate = np.ones(len(output_investigate), dtype=np.int64)
            output_trafficControl = np.ones(len(output_trafficControl), dtype=np.int64)

        else: 
            # setting masks for both policies
            amsize_investigate = len(self.action_mask(agent_name))
            new_mask_investigate = [False] * amsize_investigate
            for i in self.analyse_indexes[agent_name]:
                new_mask_investigate[i] = True
            for i in self.decoy_indexes[agent_name]:
                new_mask_investigate[i] = True

            amsize_recover = len(self.action_mask(agent_name))
            new_mask_recover = [False] * amsize_recover
            for li in self.remove_indexes[agent_name]:
                for i in self.remove_indexes[agent_name][li]:
                    new_mask_recover[i] = True
            for li in self.restore_indexes[agent_name]:
                for i in self.restore_indexes[agent_name][li]:
                    new_mask_recover[i] = True

            amsize_trafficControl = len(self.action_mask(agent_name))
            new_mask_trafficControl = [False] * amsize_trafficControl
            for li in self.block_traffic_indexes[agent_name]:
                for i in self.block_traffic_indexes[agent_name][li]:
                    new_mask_trafficControl[i] = new_mask[i]
            for li in self.allow_traffic_indexes[agent_name]:
                for i in self.allow_traffic_indexes[agent_name][li]:
                    new_mask_trafficControl[i] = new_mask[i]

            #print("\nnew_mask_trafficControl", new_mask_trafficControl)

            if iocs_exist:
                self.high_actionmask[agent_name] = np.array([0,1,0], dtype=np.int64)
                self.return_agent[agent_name] = agent_name+"_recover"
            #elif sum(network_iocs_subvector) > 0:
            #    self.high_actionmask[agent_name] = np.array([0,0,1], dtype=np.int64)
            #    self.return_agent[agent_name] = agent_name+"_trafficControl"
            else:
                #self.high_actionmask[agent_name] = np.array([1,0,0], dtype=np.int64)
                #self.return_agent[agent_name] = agent_name+"_investigate"
                
                rr = self.np_random.random()
                if rr <= 0.75:
                    self.high_actionmask[agent_name] = np.array([1,0,0], dtype=np.int64)
                    self.return_agent[agent_name] = agent_name+"_investigate"
                else:
                    self.high_actionmask[agent_name] = np.array([0,0,1], dtype=np.int64)
                    self.return_agent[agent_name] = agent_name+"_trafficControl"

        return output, output_investigate, output_recover, output_trafficControl, np.array(new_mask_investigate, dtype=np.int64), np.array(new_mask_recover, dtype=np.int64), np.array(new_mask_trafficControl, dtype=np.int64)


    def _encode_messages(self, agent_name):
        
        agent_name = agent_name[:12]

        # this is the message from agent_name broadcast to all other agents
        res_message = np.zeros(MESSAGE_LENGTH, dtype=bool)

        # we have 8 bits available 
        # we encode the subnet number (1 to 9) on 4 bits and host index (0 to 15) on 4 bits
        # start the subnet indexes from 1, to avoid collision with empty message, in cases when host is also zero 

        if len(self.decoy_msg[agent_name])> 0:
            subnet_index, host_index = self.decoy_msg[agent_name].pop()

            subnet_index += 1 # shift by 1

            # convert to binary
            res_message = [bool(int(i)) for i in f'{subnet_index:04b}{host_index:04b}']
            #print('\nEncoded message', res_message, 'from', agent_name, 'compromise in', self.subnet_names[subnet_index-1])
            
            res_message = np.array(res_message, dtype=bool)

        return res_message    


    def _decode_messages(self, agent_name, msg_list):
        
        agent_name = agent_name[:12]

        # msg_list is a 4-element list, where each element is a message (array) from other agents
        # look for messages that refer to one of agent's subnets
        for msg in msg_list:
            subnet_index = int("".join(str(int(i)) for i in msg[:4]), 2)
            if subnet_index  == 0: continue # this is  an empty message, subnet indexes start from 1

            host_index = int("".join(str(int(i)) for i in msg[4:]), 2)
            subnet_index -= 1 # reverse the shift by 1
            subnet_name = self.subnet_names[subnet_index]
            #print("decoding non-empty msg", agent_name, msg, subnet_index, subnet_name, host_index)

            subnets = self.subnets(agent_name)
            if subnet_name in subnets:
                #print("\nThis message is for me!!", agent_name)
                self.decoy_access_from[agent_name][subnet_name][host_index] = True


    # tell other if you have any IOC is your subnets
    def _encode_messages_v2(self, agent_name):

        agent_name = agent_name[:12]

        # this is the message from agent_name broadcast to all other agents
        res_message = np.zeros(MESSAGE_LENGTH, dtype=bool)

        subnets = self.subnets(agent_name)
        
        for subnet in subnets:
            if sum(self.decoy_access_from[agent_name][subnet]) + sum(self.files[agent_name][subnet]) > 0:
                i = subnets_list_blue.index(subnet)
                res_message[i] = True
                #print("Encoded IOC: ", agent_name, subnet)

        res_message = np.array(res_message, dtype=bool)
        return res_message


    # find outwhat subnets have IOCs
    def _decode_messages_v2(self, agent_name, msg_list):

        agent_name = agent_name[:12]
        res = np.zeros(NUM_SUBNETS, dtype=bool)

        # mark with an 1 all subnets with an IOCs collected from the messages
        bitwiseor = reduce(lambda x, y: x | y, msg_list)

        for subnet in subnets_list_blue:
            i = self.subnet_names.index(subnet)
            j = subnets_list_blue.index(subnet)
            res[i] = bitwiseor[j]
        
        res = np.array(res, dtype=bool)

        return res
 

    def _get_state_info(self):

        # Get true state info from environment
        get_dict = {
            'Sessions': 'All'
        }
        get_dict_per_host = {host: get_dict for host in self.hostnames}
        true_state_dict = self.env.get_true_state(info=get_dict_per_host)
        true_state_dict.pop("success")

        pprint.pprint(true_state_dict)


    def _build_comms_policy(self):
        policy_dict = {}
        mission_phases = ["Preplanning", "MissionA", "MissionB"]
        for mission in mission_phases:
            network = self._build_comms_policy_network(mission)
            index = mission_phases.index(mission)
            policy_dict[index] = network
        return policy_dict

    def _build_comms_policy_network(self, mission: str):
        hosts = (
            "internet_subnet",
            "admin_network_subnet",
            "office_network_subnet",
            "public_access_zone_subnet",
            "contractor_network_subnet",
            "restricted_zone_a_subnet",
            "restricted_zone_b_subnet",
        )

        network = nx.complete_graph(len(hosts))
        node_mapping = dict(enumerate(hosts))
        network = nx.relabel_nodes(network, node_mapping)

        network.add_edges_from((
            ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
            ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
        ))

        if mission == "MissionA":
            network.remove_edges_from((
                ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
                ("restricted_zone_a_subnet", "contractor_network_subnet"),
                ("restricted_zone_a_subnet", "restricted_zone_b_subnet"),
                ("restricted_zone_a_subnet", "internet_subnet"),
            ))
        elif mission == "MissionB":
            network.remove_edges_from((
                ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
                ("restricted_zone_b_subnet", "contractor_network_subnet"),
                ("restricted_zone_b_subnet", "restricted_zone_a_subnet"),
                ("restricted_zone_b_subnet", "internet_subnet"),
            ))
        return network
    
    def set_action_mask(self, agent_name, index, val):
        super().set_action_mask(agent_name, index, val)

    def action_space(self, agent_name: str) -> Space:
        """Returns the discrete space corresponding to the given agent."""
        return super().action_space(agent_name)

    def action_mask(self, agent_name: str) -> list[bool]:
        """Returns an ordered list corresponding to whether an action is valid or not."""
        return super().action_mask(agent_name)

    def action_labels(self, agent_name: str) -> list[str]:
        """Returns an ordered list of human-readable actions."""
        return super().action_labels(agent_name)

    def get_mission_phase(self):
        state = self.env.environment_controller.state
        return state.mission_phase

    def _get_procesess(self, state: State, hostname: str):
        observed_proc_events = state.hosts[hostname].events.old_process_creation
        unobserved_proc_events = state.hosts[hostname].events.process_creation
        return observed_proc_events + unobserved_proc_events

    def _get_connections(self, state: State, hostname: str):
        observed_conn_events = state.hosts[hostname].events.old_network_connections
        unobserved_conn_events = state.hosts[hostname].events.network_connections
        return observed_conn_events + unobserved_conn_events

    # erase history for remove/restore
    def erase_history_if_recovered(self, agent_name, observation, subnet, subnet_hosts):
        if 'success' not in observation or observation['success'].name != 'TRUE': return
        if 'action' not in observation: return
        if 'Remove' not in observation['action'].name and 'Restore' not in observation['action'].name: return
        if subnet not in str(observation['action']): return

        for i in range(len(subnet_hosts)):
            h = subnet_hosts[i]
            if h in str(observation['action']):
                self.suspicious_processes[agent_name][subnet][i] = False
                self.suspicious_connections[agent_name][subnet][i] = False

                if USE_DECOYS == True:
                    self.decoy_access_from[agent_name][subnet][i] = False

                if USE_FILES == True: 
                    # files are only erased by Restore
                    # Remove only kills processes
                    # although the environment does not clear the files out from the state for Remove,
                    # we should remove them from our local history since they are stale iocs
                    # stale iocs might be a reason why the state-based version does not go higher
                    # since the action was successful, we can erase files too

                    #if 'Restore' in observation['action'].name:
                    self.files[agent_name][subnet][i] = 0
                break


    def _get_infected(self, state: State, hostname: str):
        for i in range(NUM_RED_AGENTS):
            agent_name = f"red_agent_{i}"
            sessions = state.hosts[hostname].sessions[agent_name]
            if len(sessions) > 0:  # if red has any sessions on this host
                return sessions

        return []


    # checking number files on a host using the env controller state
    def _get_files_from_state(self, state: State, hostname: str):
        return state.hosts[hostname].files


    def get_files_from_analyse(self, observation):
        if 'success' not in observation or observation['success'].name != 'TRUE': 
            return None, []
        if 'action' not in observation or 'Analyse' not in observation['action'].name:
            return None, []

        hostname = str(observation['action']).split()[1]
        if hostname not in observation:
            return  hostname, []

        if 'Files' in observation[hostname]:
            return hostname, observation[hostname]['Files']

        return hostname, []


    def _get_decoys(self, state: State, hostname: str):
        processes = state.hosts[hostname].processes
        decoys = set() # the set of ports that have a decoy

        for p in processes:
            if p.decoy_type.name == 'EXPLOIT':
                decoys.add(p.open_ports[0]['local_port'])
        return decoys


    def update_who_accessed_decoys(self, agent_name, hostname):
        # find out who (what remote ip address) accessed decoys on hostname
        # we may want to maintain deployed decoys in memory instead of querying the state

        state = self.env.environment_controller.state
        conns = self._get_connections(state, hostname)
        if len(conns) == 0:
            return

        decoy_ports = self._get_decoys(state, hostname)

        if len(decoy_ports) == 0:
            #if len(conns) >  0:
            #    print("no decoys on ", hostname)
            return

        for c in conns:
            
            # red exploits based on port
            if c.local_port not in decoy_ports: 
                continue
                
            #print("accessing decoys:", c.local_port, 'from', c.remote_address, 'to', c.local_address, "decoy ports", decoy_ports)

            # originator of this connection is malicious
            ip = c.remote_address
            h = self.env.environment_controller.state.ip_addresses[ip]

            #  find the agent and subnet where this malicious ip belongs
            for a in self.agents:
                if h not in self.hosts(a): continue

                for subnet in self.subnets(a):
                    if subnet not in h: continue #  the subnet name  is in the host name
                        
                    subnet_hosts = [sh for sh in self.hosts(a) if subnet in sh and "router" not in sh]
                    i = subnet_hosts.index(h)

                    # if decoy access comes from its own subnets
                    # update self.decoy_access_from subvector
                    if  a == agent_name:
                        self.decoy_access_from[agent_name][subnet][i] = True
                   
                    # otherwise prepare info to send to other agents through messages
                    # agent_name identifies who sends the message
                    # subnet (and i) identifies where  the compromise is located
                    else:
                        subnet_index = self.subnet_names.index(subnet)
                        self.decoy_msg[agent_name].add((subnet_index, i))
                        
                    break

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name: str) -> Space:
        """Returns the multi-discrete space corresponding to the given agent."""
        return self._observation_space[agent_name]

    @functools.lru_cache(maxsize=None)
    def observation_spaces(self) -> dict[str, Space]:
        """Returns multi-discrete spaces corresponding to each agent."""
        return {a: self.observation_space(a) for a in self.possible_agents}
