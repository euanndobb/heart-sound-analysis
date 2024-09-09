import functools
import itertools
import operator
from typing import List
import pandas as pd
import numpy as np
import pathlib

import torch

SEGMENTATION_FS = 100  # Hz
SEGMENTATION_STATES = ('S1', 'systole', 'S2', 'diastole', 'murmur')


@functools.lru_cache(maxsize=1)
def load_segmenter(model_file: str):
    model = torch.jit.load(model_file)
    return model


def calculate_segmentation_confidence(posteriors: torch.Tensor, segmentation: torch.Tensor):
    return posteriors[segmentation, torch.arange(len(segmentation))].mean()


def get_states_format(segmentation: torch.Tensor, states: List[str] = SEGMENTATION_STATES):
    sound_sections = [
        (state, [i for i, value in it])
        for state, it in itertools.groupby(enumerate(segmentation), key=operator.itemgetter(1))
    ]

    # Get the start and end indices of each section
    sound_start_end = [(states[state], section[0], section[-1])
                       for state, section in sound_sections]

    return pd.DataFrame(sound_start_end, columns=("state", "start", "end"))


def load_default_segmenter():
    model_file = pathlib.Path(__file__).resolve().parent / "269_segmenter.pt"
    return load_segmenter(str(model_file))


def predict_neural_net(x, fs):
    x = torch.as_tensor(x)
    posteriors, segmentation = load_default_segmenter()(x, fs)

    # Remove leading batch dimension
    posteriors = posteriors.squeeze(0)
    segmentation = segmentation.squeeze(0)

    confidence = float(calculate_segmentation_confidence(posteriors, segmentation))
    return posteriors, segmentation, confidence


def resample_segmentation(x, fs_in, fs_out, sig_len):
    states_df = get_states_format(x, SEGMENTATION_STATES)
    states_df["start"] = (states_df["start"] * (fs_out / fs_in)).astype(int)
    states_df["end"] = (states_df["start"].shift(-1) - 1)
    states_df.at[states_df.index[-1], "end"] = sig_len
    states_df["end"] = states_df["end"].astype(int)

    new_segmentation = -1 * np.ones(sig_len, dtype=int)
    for _, row in states_df.iterrows():
        new_segmentation[row["start"]: row["end"] + 1] = SEGMENTATION_STATES.index(row["state"])

    if (new_segmentation == -1).any():
        raise ValueError("Unfilled values in resample segmentation")

    return new_segmentation
