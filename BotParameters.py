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
        # Return 8 actions, type=1 for continuous
        return self._n_controller_inputs, 1

    def reset(self, agents, state, shared_info):
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]):
        parsed_actions = {}
        for agent, action in actions.items():
            car_controls = np.zeros((1, self._n_controller_inputs), dtype=np.float32)
            car_controls[0, :] = action[:]
            # Last 3 are binary
            car_controls[0, -3:] = np.round((car_controls[0, -3:] + 1) / 2)
            parsed_actions[agent] = car_controls
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