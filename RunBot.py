from rlbot.utils.start_rlbot import run_bot
from bot import MyBot  # Make sure this points to your PPO bot class

# === Configure your bots ===
# Team: 0 = Blue, 1 = Orange
# Index: 0, 1, 2... unique per car

# Example: 1 bot on Blue
run_bot(MyBot, team=0, index=0)

# If you want another bot on Orange:
# run_bot(MyBot, team=1, index=1)