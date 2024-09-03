class FrameControl:
    def __init__(self, frequency: int) -> None:
        """
        frequency: number of frames in one second
        """
        self._freq = frequency
        self._last_time = 0

    def check(self, timestamp: float):
        """
        timestamp: timestamp in ms
        """
        new_time = timestamp
        last_time = self._last_time
        if (new_time - last_time) < (1e3 / self._freq):
            return False
        else:
            self._last_time = new_time
            return True
