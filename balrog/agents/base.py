class BaseAgent:
    """Base class for agents using prompt-based interactions."""

    def __init__(self, client_factory, prompt_builder):
        """Initialize the agent with a client and prompt builder."""
        self.client = client_factory()
        self.prompt_builder = prompt_builder

    def act(self, obs):
        """Generate an action based on the observation."""
        raise NotImplementedError

    def update_prompt(self, observation, action):
        """Update the prompt with the observation and action."""
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def build_system_prompt(self, env_instruction: str) -> str:
        """Build the system prompt, optionally modifying the environment instruction.

        Override this method to prepend/append agent-specific instructions.
        Default implementation returns the environment instruction unchanged.
        """
        return env_instruction

    def reset(self):
        """Reset the prompt builder."""
        self.prompt_builder.reset()
