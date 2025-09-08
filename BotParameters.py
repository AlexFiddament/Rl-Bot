from rlgym.utils.reward_functions import RewardFunction
import numpy as np

class GroundedReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        reward = 0.0

        # Reward being close to the ball
        ball_dist = np.linalg.norm(player.car_data.position - state.ball.position)
        reward += 1 / (ball_dist + 1)

        # Encourage staying on the ground
        if player.car_data.position[2] < 20:  # close to ground
            reward += 0.1
        else:
            reward -= 0.2  # discourage jumping too much

        # Reward touching the ball
        if player.ball_touched:
            reward += 1.0

        return reward
