from typing import Optional, List, Callable, Literal, Union, Tuple

import torch
from torch import Tensor, LongTensor
from scipy.stats import ks_2samp
from scipy.spatial.distance import euclidean


def categorical_vector_to_count_vector(t: Tensor, num_categories: int) -> Tensor:
    return torch.bincount(t, minlength=num_categories)


def discrete_goodness_of_fit_test(
    real: Tensor,
    synthetic: Tensor,
    num_categories: Optional[int] = None,
    make_counts: bool = True,
) -> float:
    if make_counts:
        real = categorical_vector_to_count_vector(real, num_categories)
        synthetic = categorical_vector_to_count_vector(synthetic, num_categories)
    return 1 / (1 + euclidean(real, synthetic))


def set_diff_1d(t1, t2, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.

    from: https://stackoverflow.com/a/72898627
    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]


def msas(
    real_temporal_data: Tensor,
    synthetic_temporal_data: Tensor,
    statistics_functions: List[Callable[[Tensor], float]],
    discrete_temporal_features_indices: Optional[LongTensor] = None,
    discrete_temporal_features_num_categories: Optional[LongTensor] = None,
    real_static_data: Optional[Tensor] = None,
    synthetic_static_data: Optional[Tensor] = None,
    discrete_static_features_indices: Optional[LongTensor] = None,
    discrete_static_features_num_categories: Optional[LongTensor] = None,
    static_data_weight: float = 0.5,
    removing_padding: bool = True,
    padding_value: float = 0.0,
    reduction: Union[Literal["mean", "sum"], None] = "mean",
) -> Union[Tensor, Tuple[Tensor, ...]]:
    # Shape tests

    assert real_temporal_data.size() == synthetic_temporal_data.size()
    if real_static_data is not None and synthetic_static_data is not None:
        assert real_static_data.size() == synthetic_static_data.size()
        assert (
            real_temporal_data.size(0)
            == real_static_data.size(0)
            == synthetic_static_data.size(0)
        )

    # temporal statistics
    temporal_ks_values = torch.zeros(
        (
            len(statistics_functions),
            real_temporal_data.size(-1)
            - (
                discrete_temporal_features_indices.size(0)
                if discrete_temporal_features_indices is not None
                else 0
            ),
        ),
        dtype=torch.float32,
    )

    # Discrete Goodness of Fit values
    temporal_discrete_gof_values = (
        torch.zeros(discrete_temporal_features_indices.size(0))
        if discrete_temporal_features_indices is not None
        else None
    )

    temporal_continuous_columns = (
        set_diff_1d(
            torch.arange(0, real_temporal_data.size(-1)),
            discrete_temporal_features_indices,
        )
        if discrete_temporal_features_indices is not None
        else torch.arange(0, real_temporal_data.size(-1))
    )

    for i, column_idx in enumerate(temporal_continuous_columns):
        real_temporal_statistics = torch.zeros(
            (len(statistics_functions), real_temporal_data.size(0)), dtype=torch.float32
        )
        synthetic_temporal_statistics = torch.zeros(
            (len(statistics_functions), synthetic_temporal_data.size(0)),
            dtype=torch.float32,
        )

        for datapoint_idx in range(real_temporal_data.size(0)):
            real_sequence = real_temporal_data[datapoint_idx]
            synthetic_sequence = synthetic_temporal_data[datapoint_idx]

            if removing_padding:
                real_sequence = real_sequence[(real_sequence == padding_value).sum(dim=1) != real_sequence.size(-1)]
                synthetic_sequence = synthetic_sequence[
                    (synthetic_sequence == padding_value).sum(dim=1) != synthetic_sequence.size(-1)
                ]

            real_sequence = real_sequence[:, column_idx]
            synthetic_sequence = synthetic_sequence[:, column_idx]

            for statistics_idx, statistic_f in enumerate(statistics_functions):
                real_temporal_statistics[statistics_idx, datapoint_idx] = statistic_f(
                    real_sequence
                )
                synthetic_temporal_statistics[
                    statistics_idx, datapoint_idx
                ] = statistic_f(synthetic_sequence)

        for statistics_idx in range(len(statistics_functions)):
            temporal_ks_values[statistics_idx, i] = (
                1
                - ks_2samp(
                    real_temporal_statistics[statistics_idx, :],
                    synthetic_temporal_statistics[statistics_idx, :],
                ).statistic
            )

    if discrete_temporal_features_indices is not None:
        for i, column_idx in enumerate(discrete_temporal_features_indices):
            counts_real = torch.zeros(
                (
                    real_temporal_data.size(1),
                    discrete_temporal_features_num_categories[i],
                ),
                dtype=torch.long,
            )
            counts_synthetic = torch.zeros(
                (
                    real_temporal_data.size(1),
                    discrete_temporal_features_num_categories[i],
                ),
                dtype=torch.long,
            )

            for datapoint_idx in range(real_temporal_data.size(0)):
                real_sequence = real_temporal_data[datapoint_idx]
                synthetic_sequence = synthetic_temporal_data[datapoint_idx]

                if removing_padding:

                    real_sequence = real_sequence[(real_sequence == padding_value).sum(dim=1) != real_sequence.size(-1)]
                    synthetic_sequence = synthetic_sequence[
                        (synthetic_sequence == padding_value).sum(dim=1) != synthetic_sequence.size(-1)
                    ]

                real_sequence = real_sequence[:, column_idx]
                synthetic_sequence = synthetic_sequence[:, column_idx]

                for timestep in range(real_sequence.size(0)):
                    counts_real[timestep, real_sequence[timestep].long()] += 1
                    counts_synthetic[timestep, synthetic_sequence[timestep].long()] += 1

            temporal_discrete_gof_values[i] = torch.FloatTensor(
                list(
                    map(
                        lambda timestep: discrete_goodness_of_fit_test(
                            counts_real[timestep],
                            counts_synthetic[timestep],
                            make_counts=False,
                        ),
                        range(counts_real.size(0)),
                    )
                )
            ).mean()

    # static statistics
    if real_static_data is not None and synthetic_static_data is not None:
        static_statistics = torch.zeros(real_static_data.size(-1), dtype=torch.float32)
        static_continuous_columns = (
            set_diff_1d(
                torch.arange(0, real_static_data.size(-1)),
                discrete_static_features_indices,
            )
            if discrete_static_features_indices is not None
            else torch.arange(0, real_static_data.size(-1))
        )

        # Continuous variables
        for column_idx in static_continuous_columns:
            static_statistics[column_idx] = (
                1
                - ks_2samp(
                    real_static_data[:, column_idx],
                    synthetic_static_data[:, column_idx],
                ).statistic
            )

        if discrete_static_features_indices is not None:
            for i, column_idx in enumerate(discrete_static_features_indices):
                static_statistics[column_idx] = discrete_goodness_of_fit_test(
                    real_static_data[:, column_idx].long(),
                    synthetic_static_data[:, column_idx].long(),
                    discrete_static_features_num_categories[i],
                )
    else:
        static_statistics = None

    if reduction == "mean":
        sim = temporal_ks_values.mean(dim=1)

        if temporal_discrete_gof_values is not None:
            sim = torch.concat((sim, temporal_discrete_gof_values)).mean()
        else:
            sim = sim.mean()

        if static_statistics is not None:
            sim = (
                1 - static_data_weight
            ) * sim + static_data_weight * static_statistics.mean()

        return sim

    elif reduction == "sum":
        sim = temporal_ks_values.sum()

        if temporal_discrete_gof_values is not None:
            sim += temporal_discrete_gof_values.sum()

        if static_statistics is not None:
            sim = (
                1 - static_data_weight
            ) * sim + static_data_weight * static_statistics.sum()

        return sim

    elif reduction is None:
        return temporal_ks_values, temporal_discrete_gof_values, static_statistics
    else:
        raise ValueError("Invalid reduction value.")
