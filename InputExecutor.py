"""The single gate between the bot's brain and the game pad.

Session plans a button sequence for every own turn and hands it here.
In 'suggest' mode — the default, and the only mode anything currently
constructs — the plan is just recorded: the dry-run loop over recorded
captures exercises read -> track -> search -> plan end to end without
a single button existing. 'act' mode forwards each plan to a backend
(VirtualGamepad) and must be requested explicitly, together with the
backend to use; nothing in the codebase does so yet.
"""


class InputExecutor:
    def __init__(self, backend=None, mode='suggest'):
        """
        Args:
            backend: object with press_sequence(buttons); required for
                (and only used in) act mode.
            mode: 'suggest' records plans without touching any input;
                'act' also presses them.
        """
        if mode not in ('suggest', 'act'):
            raise ValueError(f"unknown executor mode {mode!r}")
        if mode == 'act' and backend is None:
            raise ValueError("act mode needs an input backend")
        self.backend = backend
        self.mode = mode
        # Every plan ever handed in, acted on or not — the dry-run
        # transcript the replay fixtures compare against.
        self.history = []

    def execute(self, plan):
        """Record the plan; press it only in act mode.

        Returns:
            True when the buttons were physically sent.
        """
        self.history.append(list(plan))
        if self.mode == 'act':
            self.backend.press_sequence(plan)
            return True
        return False
