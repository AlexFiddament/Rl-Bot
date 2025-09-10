

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
    import random
    import numpy as np
    from BotParameters import ContinuousRLGymWrapper

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
        (InAirReward(), 0.00),
        (SpeedTowardBallReward(), 0.1),
        (VelocityBallToGoalReward(), 0.05),
        (GoalReward(), 100.0)
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




    return ContinuousRLGymWrapper(rlgym_env)



if __name__ == "__main__":
    from rlgym_ppo import Learner
    import numpy as np



    n_proc = 20
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01,
        policy_lr=1e-5,
        critic_lr=1e-5,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=True,
        save_every_ts=1_000_000,
        timestep_limit=1_000_000_000,
        log_to_wandb=True




    )

    try:
        learner.learn()  # your build_rlgym_v2_env() is called internally

    except KeyboardInterrupt:
        print("Training stopped by user")


