"""
Hosts sampler mixin (used by agent to add create dataset functionality)
"""
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
from banditbench.tasks.env import VerbalBandit, Bandit
from banditbench.tasks.cb.env import ContextualBandit
from banditbench.tasks.types import Trajectory

from banditbench.sampling.types import DatasetBuffer

import concurrent.futures
from tqdm import tqdm

from copy import deepcopy

def get_trajectory_seeds(env_seed, n_trajs):
    """Get seeds for environment and trajectories.
    
    Args:
        env: The environment to get seeds for
        n_trajs: Number of trajectory seeds to generate
    Returns:
        List of trajectory seeds
    """
    # Get environment seed or generate random one
    if env_seed is None:
        env_seed = np.random.randint(0, 2**32-1)
        
    # Generate trajectory seeds
    rng = np.random.RandomState(env_seed)
    return rng.randint(0, 2**32-1, size=n_trajs)

class SampleBase:

    def in_context_learn(self, env: Union[Bandit, ContextualBandit], n_trajs=20, *args, **kwargs) -> DatasetBuffer:
        """Collect interactions from environment and store in buffer.
        
        Args:
            env: The environment to collect from (Verbal or non-verbal)
            agent: Agent to collect data with
            n_trajs: Number of self-improving trajectories to collect
        """
        # Check if environment is verbal by looking for verbal_info property
        is_verbal = hasattr(env, 'action_names')
        is_contextual = hasattr(env, 'feature_dim')

        traj_seeds = get_trajectory_seeds(env.seed, n_trajs)
        buffer = DatasetBuffer()

        for traj_seed in tqdm(traj_seeds, desc="Collecting trajectories"):
            trajectory = []
            self.reset()

            if is_contextual:
                # Contextual bandit case
                state, _ = env.reset(seed=traj_seed)
                done = False
                while not done:
                    action = self.act(state)
                    new_state, reward, done, info = env.step(state, action)
                    trajectory.append(info['interaction'])
                    self.update(state, action, reward, info)
                    state = new_state
            else:
                # Multi-armed bandit case
                env.reset(seed=traj_seed)
                done = False
                while not done:
                    action = self.act()
                    _, reward, done, info = env.step(action)
                    trajectory.append(info['interaction'])
                    self.update(action, reward, info)

            buffer.append(Trajectory(trajectory))

        return buffer
class VerbalGuideSampleBase:
    # Using AG to collect data will produce trajectory AND fill in side-info for each action

    def in_context_learn(self, env: Union[Bandit, ContextualBandit], n_trajs=20, *args, **kwargs) -> DatasetBuffer:
        # AG has an underlying agent
        # but also provides utility class to load in action info
        # we need to both get the interaction from the underlying agent
        # and collect the action info from the AG
        is_contextual = hasattr(env, 'feature_dim')

        traj_seeds = get_trajectory_seeds(env.seed, n_trajs)
        buffer = DatasetBuffer()

        for traj_seed in tqdm(traj_seeds, desc="Collecting trajectories"):
            trajectory = []
            ag_info = []

            self.agent.reset()

            if is_contextual:
                # Contextual bandit case
                state, _ = env.reset(seed=traj_seed)
                done = False
                while not done:
                    action = self.agent.act(state)
                    new_state, reward, done, info = env.step(state, action)
                    action_info = self.get_state_actions_guide_info(state)

                    trajectory.append(info['interaction'])
                    ag_info.append(action_info)

                    self.agent.update(state, action, reward, info)
                    state = new_state
            else:
                # Multi-armed bandit case
                env.reset(seed=traj_seed)
                done = False
                while not done:
                    action = self.agent.act()
                    _, reward, done, info = env.step(action)
                    action_info = self.get_actions_guide_info()

                    trajectory.append(info['interaction'])
                    ag_info.append(action_info)

                    self.agent.update(action, reward, info)

            buffer.add(Trajectory(trajectory), ag_info)

        return buffer

class AgentSampleBase:
    # Using LLMAgent to collect data will produce trajectory, fill in side-info for each action (optional), AND fill in verbal prompt
    # will fill in side-info only if `ag` is in the LLM Agent

    def in_context_learn(self, env: VerbalBandit, n_trajs=20, num_threads=10) -> DatasetBuffer:
        
        is_contextual = hasattr(env.core_bandit, 'feature_dim')
        buffer = DatasetBuffer()
        
        # Get trajectory seeds upfront
        traj_seeds = get_trajectory_seeds(env.core_bandit.seed, n_trajs)

        def collect_single_trajectory(trial_idx, trial_seed, trial_pbar):
            trajectory = []
            ag_info = []
            verbal_prompts = []
            
            # Create a new instance for thread safety
            env_copy = deepcopy(env)
            agent_copy = deepcopy(self)
            agent_copy.reset()

            # Set position to ensure proper display of nested progress bars
            step_pbar = tqdm(total=env_copy.horizon, 
                            desc=f'Trial {trial_idx+1} steps', 
                            leave=False,
                            position=trial_idx + 1)

            if is_contextual:
                # Contextual bandit case
                state, _ = env_copy.reset(seed=trial_seed)
                done = False
                while not done:
                    # Get verbal prompts for this step
                    task_instruction = agent_copy.get_task_instruction()
                    action_history = agent_copy.get_action_history()
                    decision_query = agent_copy.get_decision_query(state)

                    action_verbal = agent_copy.act(state)
                    verbal_prompts.append({
                        'task_instruction': task_instruction,
                        'action_history': action_history,
                        'decision_query': decision_query,
                        'label': action_verbal
                    })
                    new_state, reward, done, info = env_copy.step(state, action_verbal)
                    if hasattr(agent_copy, 'ag'):
                        action_info = agent_copy.ag.get_state_actions_guide_info(state)
                        ag_info.append(action_info)

                    trajectory.append(info['interaction'])
                    action = info['interaction'].mapped_action

                    agent_copy.update(state, action, reward, info)
                    state = new_state
                    step_pbar.update(1)
            else:
                # Multi-armed bandit case  
                env_copy.reset(seed=trial_seed)
                done = False
                while not done:
                    # Get verbal prompts for this step
                    task_instruction = agent_copy.get_task_instruction()
                    action_history = agent_copy.get_action_history()
                    decision_query = agent_copy.get_decision_query()

                    action_verbal = agent_copy.act()

                    verbal_prompts.append({
                        'task_instruction': task_instruction,
                        'action_history': action_history,
                        'decision_query': decision_query,
                        'label': action_verbal
                    })
                    _, reward, done, info = env_copy.step(action_verbal)
                    if hasattr(agent_copy, 'ag'):
                        action_info = agent_copy.ag.get_actions_guide_info()
                        ag_info.append(action_info)

                    trajectory.append(info['interaction'])
                    action = info['interaction'].mapped_action

                    agent_copy.update(action, reward, info)
                    step_pbar.update(1)

            step_pbar.close()
            trial_pbar.update(1)
            return Trajectory(trajectory), ag_info if hasattr(agent_copy, 'ag') else None, verbal_prompts

        # Update the main progress bar
        if num_threads > 1 and n_trajs > 1:
            with tqdm(total=n_trajs, desc="Collecting trajectories", position=0) as trial_pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(collect_single_trajectory, i, traj_seeds[i], trial_pbar)
                              for i in range(n_trajs)]

                    for future in concurrent.futures.as_completed(futures):
                        trajectory, ag_info, verbal_prompts = future.result()
                        buffer.add(trajectory, ag_info, verbal_prompts)
        else:
            with tqdm(total=n_trajs, desc="Collecting trajectories", position=0) as trial_pbar:
                for i in range(n_trajs):
                    trajectory, ag_info, verbal_prompts = collect_single_trajectory(i, traj_seeds[i], trial_pbar)
                    buffer.add(trajectory, ag_info, verbal_prompts)

        return buffer
