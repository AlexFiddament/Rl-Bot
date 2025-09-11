import asyncio
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

class MyBot(BaseAgent):
    def initialize_agent(self):
        print(f"{self.name} initialized!")

    async def run(self):
        while True:
            controls = SimpleControllerState()
            controls.throttle = 1.0  # drives forward
            self.set_controller_state(controls)

            await asyncio.sleep(0.016)  # ~60 FPS

