from typing import Optional, List, Callable, Literal, Union, Tuple

import torch
from torch import Tensor, LongTensor
from scipy.stats import ks_2samp
from scipy.spatial.distance import euclidean


def categorical_vector_to_freq_vector(t: Tensor, num_categories: int) -> Tensor:
    """
    Converts a categorical vector to a frequency vector.

    Args:
        `t` (Tensor): A 1D tensor representing the categorical vector.

        `num_categories` (int): The number of possible categories the variable can take.

    Returns:
        Tensor: A 1D tensor representing the frequency vector.
    """
    tc = torch.bincount(t, minlength=num_categories)
    return tc / tc.sum()


def discrete_goodness_of_fit_test(
    real: Tensor,
    synthetic: Tensor,
    num_categories: Optional[int] = None,
    make_counts: bool = True,
) -> float:
    """
    Computes the goodness-of-fit between two discrete distributions using the Euclidean
    distance.

    Args:
        `real` (Tensor): A 1D tensor representing the real distribution or frequency
            vectors.
        `synthetic` (Tensor): A 1D tensor representing the synthetic distribution or
            frequency vectors.
        `num_categories` (Optional[int], optional): The number of categories in the
            distribution. If None, `real` and `synthetic` are assumed to be frequency
            vectors. Defaults to None.
        `make_counts` (bool, optional): Whether to convert the input vectors to
            frequency vectors. If True, the input vectors are assumed to be categorical
            vectors and are converted to frequency vectors. If False,
            the input vectors are assumed to be frequency vectors. Defaults to True.

    Returns:
        float: The goodness-of-fit between the two distributions,
            computed as 1 minus the Euclidean distance between the two frequency
            vectors normalized by sqrt(2) -- maximum theoretical distance.
    """
    if make_counts:
        real = categorical_vector_to_freq_vector(real, num_categories)
        synthetic = categorical_vector_to_freq_vector(synthetic, num_categories)

    # maximum euclidean between two frequency vectors is sqrt(2)
    # we want this to be bounded between 0 and 1
    return 1 - euclidean(real, synthetic) / (2 ** 0.5)


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
    statistics_functions: List[Callable[[Tensor], Union[float, Tensor]]],
    discrete_temporal_features_indices: Optional[LongTensor] = None,
    discrete_temporal_features_num_categories: Optional[LongTensor] = None,
    real_static_data: Optional[Tensor] = None,
    synthetic_static_data: Optional[Tensor] = None,
    discrete_static_features_indices: Optional[LongTensor] = None,
    discrete_static_features_num_categories: Optional[LongTensor] = None,
    static_data_weight: float = 0.5,
    removing_padding: bool = True,
    padding_value: float = 0.0,
    ignore_nans: bool = True,
    reduction: Union[Literal["mean", "sum"], None] = "mean",
    enforce_temporal_shape: bool = True,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """
    Computes the MSAS (Multi-Sequence Aggregate Similarity) between two
    sets of temporal data, and optionally two sets of static data.

    Args:
        `real_temporal_data` (Tensor): A tensor of shape (N, L, C) representing the real
            temporal data, where N is the number of data points, L is the length of the
            temporal sequence, and C is the number of features (continuous or discrete).
        `synthetic_temporal_data` (Tensor): A tensor of shape (N, L, C) representing the
            synthetic temporal data, where N is the number of data points, L is the
            length of the temporal sequence, and C is the number of features (continuous
            or discrete).
        `statistics_functions` (List[Callable[[Tensor], float]]): A list of functions
            that compute statistics on a 1D tensor (e.g., mean, variance, skewness,
            kurtosis, etc.). These functions will be used to compute the goodness-of-fit
            values between the real and synthetic temporal data.
        `discrete_temporal_features_indices` (Optional[LongTensor]): A tensor of shape
            (D,) representing the indices of the discrete features in the temporal data,
            where D is the number of discrete features. If None, all features are
            assumed to be continuous.
        `discrete_temporal_features_num_categories` (Optional[LongTensor]): A tensor of
            shape (D,) representing the number of categories for each discrete feature
            in the temporal data. If None, all features are assumed to be continuous.
        `real_static_data` (Optional[Tensor]): A tensor of shape (N, S) representing
            the real static data, where N is the number of data points, and S is the
            number of static features (continuous or discrete). If None, no static data
            will be used.
        `synthetic_static_data` (Optional[Tensor]): A tensor of shape (N, S)
            representing the synthetic static data, where N is the number of data
            points, and S is the number of static features (continuous or discrete).
            If None, no static data will be used.
        `discrete_static_features_indices` (Optional[LongTensor]): A tensor of shape
            (D',) representing the indices of the discrete features in the static data,
            where D' is the number of discrete features. If None, all features are
            assumed to be continuous.
        `discrete_static_features_num_categories` (Optional[LongTensor]): A tensor of
            shape (D',) representing the number of categories for each discrete feature
            in the static data. If None, all features are assumed to be continuous.
        `static_data_weight` (float): A weight between 0 and 1 that determines the
            relative importance of the temporal and static similarities in the MSAS
            score. A value of 0 means that only the temporal score is used, while a
            value of 1 means that only the static data is used.
        `removing_padding` (bool): A flag that determines whether to remove padded
            temporal timesteps from the temporal data before computing the statistics.
            If True, any sequence that consists entirely of padding values will be
            removed. Default is True.
        `padding_value` (float): A value that represents padding in the temporal data.
            Default is 0.0.
        `ignore_nans` (bool): A flag that determines whether to ignore NaN values during
            the computation of the temporal statistics. This may happen when, as an example,
            computing the standard deviation of a sequence with a single value. Default is True.
        `reduction` (Union[Literal["mean", "sum"], None]): A string that determines how
            to reduce the MSAS score across statistics function and static similarity.
            If "mean", the mean score is returned. If "sum", the sum score is
            returned. If None, a tuple of tensors is returned. Default is "mean".
        `enforce_temporal_shape` (bool): A flag that determines whether to enforce that
            the real and synthetic temporal data have the same shape. Default is True.

    Returns:
        Union[Tensor, Tuple[Tensor, ...]]: The MSAS score(s) between the real and
        synthetic data, and optionally the real and synthetic static data.
        If reduction is "mean" or "sum", a single tensor is returned. If
        reduction is None, a tuple of tensors is returned. The tuple contains three
        tensors: the Kolmogorov-Smirnov statistics for each statistic function in the
        temporal data, the discrete goodness-of-fit values for each discrete feature in
        the temporal data, and the static similarity between the real and synthetic
        data.
    """
    # Shape tests
    if enforce_temporal_shape:
        try:
            assert real_temporal_data.size() == synthetic_temporal_data.size()
        except AssertionError:
            raise ValueError(
                f"Real temporal shape {real_temporal_data.size()} and synthetic temporal shape {synthetic_temporal_data.size()} do not match."
            )
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
                real_sequence = real_sequence[
                    (real_sequence == padding_value).sum(dim=1)
                    != real_sequence.size(-1)
                ]
                synthetic_sequence = synthetic_sequence[
                    (synthetic_sequence == padding_value).sum(dim=1)
                    != synthetic_sequence.size(-1)
                ]

            real_sequence = real_sequence[:, column_idx]
            synthetic_sequence = synthetic_sequence[:, column_idx]

            for statistics_idx, statistic_f in enumerate(statistics_functions):
                real_stat = statistic_f(real_sequence)
                synthetic_stat = statistic_f(synthetic_sequence)

                if torch.isnan(real_stat) or torch.isnan(synthetic_stat):
                    if ignore_nans:
                        continue
                    else:
                        error_source = "real" if torch.isnan(real_stat) else "synthetic"
                        raise ValueError(
                            f"Statistic function {statistic_f} returned NaN value for datapoint index {datapoint_idx} in the {error_source} dataset."
                        )

                real_temporal_statistics[statistics_idx, datapoint_idx] = real_stat
                synthetic_temporal_statistics[statistics_idx, datapoint_idx] = (
                    synthetic_stat
                )

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
                    synthetic_temporal_data.size(1),
                    discrete_temporal_features_num_categories[i],
                ),
                dtype=torch.long,
            )

            for datapoint_idx in range(real_temporal_data.size(0)):
                real_sequence = real_temporal_data[datapoint_idx]
                synthetic_sequence = synthetic_temporal_data[datapoint_idx]

                if removing_padding:
                    real_sequence = real_sequence[
                        (real_sequence == padding_value).sum(dim=1)
                        != real_sequence.size(-1)
                    ]
                    synthetic_sequence = synthetic_sequence[
                        (synthetic_sequence == padding_value).sum(dim=1)
                        != synthetic_sequence.size(-1)
                    ]

                real_sequence = real_sequence[:, column_idx]
                synthetic_sequence = synthetic_sequence[:, column_idx]

                for timestep in range(real_sequence.size(0)):
                    counts_real[timestep, real_sequence[timestep].long()] += 1

                for timestep in range(synthetic_sequence.size(0)):
                    counts_synthetic[timestep, synthetic_sequence[timestep].long()] += 1

            temporal_discrete_gof_values[i] = torch.FloatTensor(
                list(
                    map(
                        lambda timestep: discrete_goodness_of_fit_test(
                            counts_real[timestep] / counts_real[timestep].sum(),
                            counts_synthetic[timestep]
                            / counts_synthetic[timestep].sum(),
                            make_counts=False,
                        ),
                        range(min(counts_real.size(0), counts_synthetic.size(0))),
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
