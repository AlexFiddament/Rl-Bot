from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np
from typing import Dict, Any
from rlgym.api import ActionParser, AgentID
from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.common_values import BLUE_TEAM
from typing import List, Dict, Any
from gym import spaces
from gymnasium.spaces import Box


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball"""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = (ball_physics.position - car_physics.position)
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal"""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist

            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)
        return rewards


class ContinuousAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, int]):
    def __init__(self):
        super().__init__()
        self._n_controller_inputs = 8

    def get_action_space(self, agent=None):
        # Number of continuous actions, type=1
        return self._n_controller_inputs, 1

    def reset(self, agents, state, shared_info):
        pass

    def parse_actions(self, actions: dict, state, shared_info):
        parsed_actions = {}
        for agent, action in actions.items():
            # Ensure action is length 8
            action = np.array(action, dtype=np.float32).flatten()
            if action.shape[0] != self._n_controller_inputs:
                # Fill with random values if wrong shape
                action = np.random.uniform(-1, 1, self._n_controller_inputs)

            # Last 3 are binary buttons
            action[-3:] = np.round((action[-3:] + 1) / 2)

            parsed_actions[agent] = action.reshape(1, self._n_controller_inputs)  # shape (1, 8)

        return parsed_actions


            







class RichObsBuilder(ObsBuilder):
    def get_obs_space(self, agent: AgentID):
        # Total number of features in your observation
        return 'real', 31

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict):
        pass

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict) -> Dict[AgentID, np.ndarray]:
        obs = {}
        for agent in agents:
            obs[agent] = self._build_obs(agent, state)
        return obs

    def _build_obs(self, agent: AgentID, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == BLUE_TEAM:
            agent_phys = car.physics
            ball = state.ball
            opponents = [c for i, c in state.cars.items() if c.team_num != BLUE_TEAM]
        else:
            agent_phys = car.inverted_physics
            ball = state.inverted_ball
            opponents = [c.inverted_physics for i, c in state.cars.items() if c.team_num == BLUE_TEAM]

        # Ball features
        ball_features = np.concatenate([ball.position, ball.linear_velocity, ball.angular_velocity])

        # Agent car features
        agent_features = np.concatenate([
            agent_phys.position,
            agent_phys.forward,
            agent_phys.up,
            agent_phys.linear_velocity,
            agent_phys.angular_velocity,
            [car.boost_amount]  # make sure 'boost' exists in Car class
        ])

        # Opponent cars (keep full Car object)
        opponents = [c for i, c in state.cars.items() if c.team_num != BLUE_TEAM]

        # Then when building features:
        if len(opponents) > 0:
            opp_car = opponents[0]
            # Use inverted_physics if needed
            if car.team_num != BLUE_TEAM:
                opp_phys = opp_car.inverted_physics
            else:
                opp_phys = opp_car.physics

            opp_features = np.concatenate([
                opp_phys.position - agent_phys.position,  # relative position
                opp_phys.linear_velocity
            ])
        else:
            opp_features = np.zeros(5)

        return np.concatenate([ball_features, agent_features, opp_features])









class ContinuousRLGymWrapper:
    """
    Wraps an RLGymV2 environment for continuous actions and PPO usage.
    Exposes proper observation_space and action_space for SB3/RLlib.
    """
    def __init__(self, rlgym_env):
        self.rlgym_env = rlgym_env

        # Reset environment to get an example observation
        obs_example = list(self.rlgym_env.reset().values())[0]  # take first agent
        self.num_agents = len(self.rlgym_env.reward_fn.reward_fns)  # approximate, or just len(obs_dict)

        # Observation space: assume all agents have the same shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_example.shape,
            dtype=np.float32
        )

        # Action space: 8 continuous actions per agent
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment and return a stacked array of observations for all agents.
        """
        obs_dict = self.rlgym_env.reset()  # dict: agent_id -> obs
        self.agent_ids = list(obs_dict.keys())
        return np.array([obs_dict[aid] for aid in self.agent_ids], dtype=np.float32)

    def step(self, actions):
        """
        Take a step in the environment.
        `actions` is expected to be shape (num_agents, 8)
        """
        action_dict = {aid: actions[i] for i, aid in enumerate(self.agent_ids)}
        obs_dict, reward_dict, done_dict, info_dict = self.rlgym_env.step(action_dict)

        obs = np.array([obs_dict[aid] for aid in self.agent_ids], dtype=np.float32)
        rewards = np.array([reward_dict[aid] for aid in self.agent_ids], dtype=np.float32)
        dones = np.array([done_dict[aid] for aid in self.agent_ids], dtype=bool)
        done_flag = dones.any()  # True if any agent is done
        infos = [info_dict[aid] for aid in self.agent_ids]

        return obs, rewards, done_flag, infos

    def close(self):
        """
        Close the underlying environment if it has a close method.
        """
        if hasattr(self.rlgym_env, "close"):
            self.rlgym_env.close()



