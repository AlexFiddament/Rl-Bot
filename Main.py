

def build_rlgym_v2_env():
    from  BotParameters import InAirReward
    from BotParameters import VelocityBallToGoalReward
    from BotParameters import SpeedTowardBallReward
    from BotParameters import ContinuousAction
    from BotParameters import RichObsBuilder
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = ContinuousAction()
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(
        (InAirReward(), 0.002),
        (SpeedTowardBallReward(), 0.01),
        (VelocityBallToGoalReward(), 0.1),
        (GoalReward(), 10.0)
    )

    obs_builder = RichObsBuilder()

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine()
    )

    obs = rlgym_env.reset()

    first_agent = list(obs.keys())[0]  # grab whatever agent ID exists
    print("Raw observation shape for agent:", obs[first_agent].shape)
    print("Raw observation sample:", obs[first_agent])

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":

    from rlgym_ppo import Learner


    # 32 processes
    n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None, # Leave this empty for now.
                      ppo_batch_size=100_000,  # batch size - much higher than 300K doesn't seem to help most people
                      policy_layer_sizes=[2048, 2048, 1024, 1024],  # policy network
                      critic_layer_sizes=[2048, 2048, 1024, 1024],  # critic network
                      ts_per_iteration=100_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=300_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      policy_lr=1e-4,  # policy learning rate
                      critic_lr=1e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=1_000_000,  # save every 1M steps
                      timestep_limit=1_000_000_000,  # Train for 1B steps
                      log_to_wandb=True # Set this to True if you want to use Weights & Biases for logging.
                      )
    learner.learn()

