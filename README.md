# Multi-Sequence Aggregate Similarity (MSAS), in PyTorch

msas-pytorch is a Python library for computing the similarity between two sets sequence data, optionally with associated static data. The main objective is to provide a similarity function to evaluate the performance of generative models, by comparing real data with synthetic data. It uses the PyTorch Tensor API for interoperability with generative modeling libraries.

It implements the [Multi-Sequence Aggregate Similarity](https://arxiv.org/abs/2207.14406), and further extends it in the following way:

- Allows the inclusion of static data in the similarity calculations, in the case you may use longitudinal datasets with static data, e.g. demographic data, comorbidities, and others. The user may set the relative importance of this component vs temporal component;
- Enabling different data types: features can not only be continuous, but also discrete/categorical.

The discrete goodness of fit test ($d: \mathbb{R}^{N \times C} \to [0,1]$), where $N$ is the sample size and $C$ is the possible values of the categorical/discrete feature, is defined as the following:

$$
d(x_{f_i}^{(1)}, x_{f_i}^{(2)}) = 1 - ||freq(x_{f_i}^{(1)}) - freq(x_{f_i}^{(2)})||_2
$$

where $x_{f_i}^{(1)}$ is sample one for feature $i$, which could be the real data, and x_{f_i}^{(2)} for a second sample, which could be synthetic data. $freq$ is a function that calculates the frequency of the possible values of the categorical/discrete feature, observed in the sample.

This is calculated in different ways, depending on whether it is static or temporal data. For static data, $d$ is calculated with the observed values across the different datapoints in the sample. For temporal data, since the feature occurs in all timesteps, we calculate $d$ for the distribution of values for each time step, and then average them out.

## Install

To install msas-pytorch, you can use pip:

```bash
$ pip install msas-pytorch
```

## Usage

An example on how to use msas-pytorch can be found in the `examples/example.ipynb` notebook.

Nevertheless, it would look like the following:

```python
from msas_pytorch import msas
import torch

# let's define some statistics functions

statistics_functions = [torch.mean, torch.std, torch.median, torch.max, torch.min]

real_static_data = torch.randn(1000, 5)
# feature with index 1 and 4 will be categorical with 10 possible values
real_static_data[:, 1] = torch.randint(0, 10, (1000,))
real_static_data[:, 1] = torch.randint(0, 10, (1000,))

real_temporal_data = torch.randn(1000, 10, 5)
# the same here
real_temporal_data[:, :, 1] = torch.randint(0, 10, (1000, 10))
real_temporal_data[:, :, 4] = torch.randint(0, 3, (1000, 10))


synthetic_static_data = torch.randn(1000, 5)
# feature with index 1 and 4 will be categorical with 10 possible values
synthetic_static_data[:, 1] = torch.randint(0, 10, (1000,))
synthetic_static_data[:, 1] = torch.randint(0, 10, (1000,))

synthetic_temporal_data = torch.randn(1000, 10, 5)
# the same here
synthetic_temporal_data[:, :, 1] = torch.randint(0, 10, (1000, 10))
# but this time this feature has only 3 possible values
synthetic_temporal_data[:, :, 4] = torch.randint(0, 3, (1000, 3))

print(
    msas(
        real_temporal_data,
        synthetic_temporal_data,
        statistics_functions
        discrete_temporal_features_indices=torch.LongTensor([1, 4]),
        discrete_temporal_features_num_categories=torch.LongTensor([10, 3]),
        real_static_data=real_static_data,
        synthetic_static_data=synthetic_static_data,
        discrete_static_features_indices=torch.LongTensor([1, 4]),
        discrete_static_features_num_categories=torch.LongTensor([10, 10]),
    )
)
```

## Contributing

Contributions to msas-pytorch are welcome! If you would like to contribute, please open an issue or pull request on the GitHub repository.

## Citations

```bibtex
@article{zhang2022sequential,
  title={Sequential Models in the Synthetic Data Vault},
  author={Zhang, Kevin and Patki, Neha and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:2207.14406},
  year={2022}
}
```