from math import isnan
import numpy as np


class PositionFilter:
    def __init__(self, dim: int, pool_size: int) -> None:
        """
        dim: the dimension of position
        pool_size: the mean filter pool size
        """
        self._pool_size = pool_size
        self._pool = np.zeros((dim, pool_size), dtype=np.float32)
        self._frame_idx = 0

    def update(self, new_pos: np.ndarray) -> np.ndarray:
        """
        new_pos: input the pos to get the filtered pos
        -> : the filtered pos
        """
        if not np.any(np.isnan(new_pos)):
            if self._frame_idx == 0:
                self._pool[:, :] = new_pos.reshape(-1, 1)
            else:
                self._pool[:, self._frame_idx % self._pool_size] = new_pos

        self._frame_idx += 1

        filtered_pos = np.mean(self._pool, axis=1)
        return filtered_pos
