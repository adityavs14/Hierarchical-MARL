from statistics import mean, stdev

from CybORG import CybORG  # the original env
from EnterpriseMAE_CC4 import EnterpriseMAE   # the wrapped env


class Metrics:

    def __init__(self, env, *args, **kwargs):
        
        self.reset_metrics(env)

        # collected for each episode, will average at the end of the game
        self.stats = {}

        # network posture
        self.stats["clean_network_frac"] = []
        self.stats["nonescalated_network_frac"] = []

        # efficiency of recoveries
        self.stats["true_positives"] = []
        self.stats["false_positives"] = []
        self.stats["recovery_precision"] = []
        self.stats["recovery_err"] = []
        self.stats["mean_time_to_recover"] = []

        # operational impact
        self.stats["impact_count"] = []
 

    def reset_metrics(self, env):
       
        # reset for each episode
        self.compromised_status_per_host = {} # hostname -> [has_red_session, has_red_privileged_session]
        state = env.environment_controller.state
        for h in state.hosts:
            if "router" in h: continue
            self.compromised_status_per_host[h] = [False, False]

        # network-level metrics, except the "contractor" subnet
        self.compromised_count = []
        self.noncompromised_count = []
        self.escalated_count = []
        self.nonescalated_count = []
        self.clean_frac = []
        self.nonescalated_frac = []

        # true/false positives metric, collected throughout each episode, for the entire blue team
        # it can be broken down by blue agent or subnet if necessary
        self.false_pos = 0
        self.true_pos = 0

        # total number of impact actions within both operational subnets during an episode
        self.impact_count = 0

        # mean time to recover
        # current stride length per host
        self.compromised_length = {} 
        self.privileged_length = {}

        # list of stride lengths within this episode
        self.compromised_lengths_list = []
        self.privileged_lengths_list = []
        

    def check_if_compromised(self, env, hostname: str):
        # check if  host is compromised, and return type of red sessions present
        
        has_session = False
        has_privileged_session = False

        state = env.environment_controller.state
        NUM_RED_AGENTS = 6
        
        for i in range(NUM_RED_AGENTS):
            agent_name = f"red_agent_{i}"

            # sessions of this red agent
            agent_sessions = state.sessions[agent_name]
            if len(agent_sessions) == 0: continue
            
            # check if this red agents holds sessions on this host
            session_ids = state.hosts[hostname].sessions[agent_name]
            if len(session_ids) ==  0: continue

            # get the type of red session user or root (privileged)
            for sid in session_ids:
                s = agent_sessions[sid]
                if s.active:
                    has_session = True 
                    if s.has_privileged_access():
                        has_privileged_session = True
                        break

        return has_session, has_privileged_session

    
    def compromise_status_per_host(self, env):
        # get compromise status for each host in the network

        compromise_status = {}
        state = env.environment_controller.state
        
        for h in state.hosts:
            if "router" in h: continue
            has_session, has_privileged_session= self.check_if_compromised(env, h)
            compromise_status[h] = [has_session, has_privileged_session]

        return compromise_status
    

    def compromise_counts(self, compromised_status_per_host):
        compromised = 0
        noncompromised = 0
        privileged = 0
        nonprivileged = 0

        for h in compromised_status_per_host:
            has_session, has_privileged_session = compromised_status_per_host[h] 

            if has_session:
                compromised += 1
            else:
                noncompromised += 1

            if has_privileged_session:
                privileged += 1
            else:
                nonprivileged += 1

        return compromised, noncompromised, privileged, nonprivileged
        

    def collect_compromise_metrics(self, compromised_status_per_host):
        # append the metrics values for this time step
        compromised, noncompromised, escalated, nonescalated = self.compromise_counts(compromised_status_per_host)

        # collect network-level metrics for this timestep
        self.compromised_count.append(compromised)
        self.noncompromised_count.append(noncompromised)
        self.escalated_count.append(escalated)
        self.nonescalated_count.append(nonescalated)
        
        # fractions
        clean_frac = noncompromised / (compromised + noncompromised)
        self.clean_frac.append(clean_frac)
        nonescalated_frac = nonescalated / (escalated + nonescalated)
        self.nonescalated_frac.append(nonescalated_frac)


    def get_compromised_status_per_host(self):
        return self.compromised_status_per_host


    def collect_true_false_pos(self, wrapped_env, actions, compromised_status_before_step):
        # collected at every timestep
        # check if a recover action is was useful:
        # restore on compromised hosts; remove on non-escalated hosts

        for agent_name in wrapped_env.agents:
            actions_list = wrapped_env.get_action_space(agent_name)['actions']
            action_index = actions[agent_name]
            action = actions_list[action_index]
            
            # only interested in remove/restore
            if 'Remove' not in action.name and 'Restore' not in action.name: continue
            
            # print(agent_name, action, "hostname", action.hostname)
            h = action.hostname
            [has_session, has_privileged_session] = compromised_status_before_step[h]

            # false positives
            if not has_session:
                self.false_pos += 1
            elif 'Remove' in action.name and has_privileged_session:
                self.false_pos += 1
            else:  # true positives
                self.true_pos += 1


    def update_impact_count(self, env):
        # impact count for the red team
        # OT services reside on every host in the operational zones
        # could we implement this based on the Impact success status instead?
        red_agents = [agent for agent in env.agents if 'red' in agent]

        for agent_name in red_agents:
            action = env.get_last_action(agent_name)[0]
            if action.name != "Impact": continue
            if "operational" not in action.hostname: continue
            self.impact_count += 1


    def update_time_to_recover(self, compromised_status_before_step, compromised_status_after_step):
        # increment the number of consecutive time steps that a hosts spends in a compromised / escalated state

        for h in compromised_status_after_step:

            compromised_after, privileged_after = compromised_status_after_step[h]
            compromised_before, privileged_before = compromised_status_before_step[h]

            if compromised_after == True:
                # host is in a compromised state (still)
                if h not in self.compromised_length:
                    self.compromised_length[h] = 0
                self.compromised_length[h] += 1 

            elif compromised_after == False and compromised_before == True:
                # host has been cleaned
                # collect this stride length to compute the  mean
                self.compromised_lengths_list.append(self.compromised_length[h])
                self.compromised_length[h] = 0

            if privileged_after == True:
                # host is in an escalated state (still)
                if h not in self.privileged_length:
                    self.privileged_length[h] = 0
                self.privileged_length[h] += 1

            elif privileged_after == False and privileged_before == True:
                # red privileged session has been cleared
                self.privileged_lengths_list.append(self.privileged_length[h])
                self.privileged_length[h] = 0


    def collect_stats_timestep(self, env, wrapped_env, actions, compromised_status_before_step):
        # this function should be called at each time step, after the step() function has been executed
        self.compromised_status_per_host = self.compromise_status_per_host(env)

        # collect count/frac of compromised, noncompromised, escalated, non-escalated hosts per subnet
        self.collect_compromise_metrics(self.compromised_status_per_host)

        # collect number of true/false positives on restore/remove for the blue team
        self.collect_true_false_pos(wrapped_env, actions, compromised_status_before_step)

        # increment the impact count for the red team
        self.update_impact_count(env)

        # update the time to recover based on the current time step
        compromised_status_after_step = self.compromised_status_per_host # current status
        self.update_time_to_recover(compromised_status_before_step, compromised_status_after_step)


    def collect_stats_episode(self):
        # stats should be collected at the end of each episode
        
        self.stats["clean_network_frac"].append(mean(self.clean_frac))
        self.stats["nonescalated_network_frac"].append(mean(self.nonescalated_frac))
        self.stats["impact_count"].append(self.impact_count)

        if self.true_pos > 0:
            self.stats["true_positives"].append(self.true_pos)

        if self.false_pos > 0:
            self.stats["false_positives"].append(self.false_pos)

        if (self.true_pos + self.false_pos) > 0:
            recovery_err = self.false_pos / (self.true_pos + self.false_pos)
            recovery_precision = 1.0 - recovery_err
            self.stats["recovery_err"].append(recovery_err)
            self.stats["recovery_precision"].append(recovery_precision)
        
        if len(self.compromised_lengths_list) > 0:
            self.stats["mean_time_to_recover"].append(mean(self.compromised_lengths_list))


    def stats_game(self):
        # averaged across all episodes
        final_stats = {}
        final_stats["clean_network_frac"] = round(mean(self.stats["clean_network_frac"]), 2)
        final_stats["nonescalated_network_frac"] = round(mean(self.stats["nonescalated_network_frac"]), 2)
        final_stats["true_positives"] = round(mean(self.stats["true_positives"]), 2)
        final_stats["false_positives"] = round(mean(self.stats["false_positives"]), 2)
        final_stats["recovery_err"] = round(mean(self.stats["recovery_err"]), 2)
        final_stats["recovery_precision"] = round(mean(self.stats["recovery_precision"]), 2)
        final_stats["mean_time_to_recover"] = round(mean(self.stats["mean_time_to_recover"]), 2)
        final_stats["impact_count"] = round(mean(self.stats["impact_count"]), 2)

        return final_stats

