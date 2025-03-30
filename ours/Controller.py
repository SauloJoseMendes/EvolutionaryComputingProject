import numpy as np


class Controller:
    def __init__(self, controller_type, action_size=None):
        self._type = controller_type
        self.action_size = action_size

    def run(self, t):
        if self._type == "alternating_gait":
            return self.alternating_gait(t)
        elif self._type == "sinusoidal_wave":
            return self.sinusoidal_wave(t)
        elif self._type == "hopping_motion":
            return self.hopping_motion(t)
        else:
            raise ValueError("Unknown controller type")
    def alternating_gait(self, t):
        """Alternates actuation to mimic a walking gait."""
        if self.action_size is None:
            raise ValueError("Action size not initialized for alternating gait controller.")
        action = np.zeros(self.action_size)

        # Alternate expansion & contraction every 10 timesteps
        if t % 20 < 10:
            action[:self.action_size // 2] = 1  # Front half expands
            action[self.action_size // 2:] = -1  # Back half contracts
        else:
            action[:self.action_size // 2] = -1  # Front half contracts
            action[self.action_size // 2:] = 1  # Back half expands

        return action

    def sinusoidal_wave(self, t):
        """Generates a wave-like motion pattern for snake-like robots."""
        if self.action_size is None:
            raise ValueError("Action size not initialized for sinusoidal wave controller.")
        action = np.zeros(self.action_size)
        for i in range(self.action_size):
            action[i] = np.sin(2 * np.pi * (t / 20 + i / self.action_size))  # Sin wave pattern
        return action

    def hopping_motion(self, t):
        """Makes the robot jump forward using periodic full-body contraction and expansion."""
        if self.action_size is None:
            raise ValueError("Action size not initialized for hopping motion controller.")
        action = np.zeros(self.action_size)
        if t % 20 < 10:
            action[:] = 1  # Expand all active voxels
        else:
            action[:] = -1  # Contract all active voxels
        return action