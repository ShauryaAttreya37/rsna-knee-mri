from __future__ import annotations

import numpy as np
import pytest

from rsna.imaging_inference import rank_average_probabilities
from rsna.imaging_training import ImagingTrainingError
from rsna.submission import TARGET_COLUMNS


def test_rank_average_normalizes_model_score_scales() -> None:
    target_count = len(TARGET_COLUMNS)
    first = np.tile(np.asarray([0.1, 0.2, 0.3])[:, None], (1, target_count))
    second = np.tile(np.asarray([0.7, 0.8, 0.9])[:, None], (1, target_count))

    result = rank_average_probabilities(np.stack([first, second]))

    assert result.shape == (3, target_count)
    assert result[:, 0] == pytest.approx([1 / 6, 0.5, 5 / 6])


def test_rank_average_handles_ties() -> None:
    values = np.full((2, 4, len(TARGET_COLUMNS)), 0.5)

    result = rank_average_probabilities(values)

    assert result == pytest.approx(np.full((4, len(TARGET_COLUMNS)), 0.5))


def test_rank_average_rejects_wrong_target_count() -> None:
    with pytest.raises(ImagingTrainingError, match="target count"):
        rank_average_probabilities(np.zeros((2, 3, 4)))
