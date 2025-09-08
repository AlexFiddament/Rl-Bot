import rlgym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Import custom reward
from BotParameters import GroundedReward



def main():
    # -----------------------
    # 1. Create Environment
    # -----------------------
    def make_env():
        # rlgym creates a Rocket League-like environment
        # we pass in our custom GroundedReward
        return rlgym.make(reward_fn=GroundedReward())

    # Stable-Baselines requires vectorized environments
    # DummyVecEnv wraps our single env to make it compatible
    env = DummyVecEnv([make_env])

    # -----------------------
    # 2. Define the SAC Model
    # -----------------------
    model = SAC(
        "MlpPolicy",       # "MlpPolicy" = multilayer perceptron (a normal neural network)
        env,               # the environment we created
        verbose=1,         # 1 = print training info, 0 = silent
        learning_rate=3e-4, # step size for network updates (default works well)
        buffer_size=1000000, # replay buffer size (stores past experiences)
        batch_size=256,     # how many samples are trained on each step
        tau=0.005,          # target smoothing coefficient (for stable Q updates)
        gamma=0.99,         # discount factor (how much future rewards matter)
        train_freq=(1, "step"),  # how often to update the network
        gradient_steps=1,   # how many backprop steps per update
        tensorboard_log="./logs/" # folder for TensorBoard training graphs
    )

    # -----------------------
    # 3. Train the Model
    # -----------------------
    model.learn(total_timesteps=100000)  # increase this for longer training

    # -----------------------
    # 4. Save the Model
    # -----------------------
    model.save("grounded_bot_sac")


if __name__ == "__main__":
    main()
