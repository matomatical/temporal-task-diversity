"""In-context linear regression setting, following Raventós et al. (2023).

Each transformer is trained to predict y_k from (x_1, y_1, ..., x_{k-1}, y_{k-1}, x_k)
under the data-generating process described in the paper's setting section.

Submodules:

- setting: transformer architecture, task and sequence distributions, losses, TrainState
- baselines: dMMSE and ridge Bayesian reference predictors
- generation: sequence rollout primitives
- predictive_montecarlo: predictive Monte Carlo for extracting the transformer's
  implicit prior over task vectors
"""

from iclr.setting import (
    DiscreteTaskDistribution,
    GaussianTaskDistribution,
    InContextRegressionTransformer,
    RegressionSequenceDistribution,
    TrainState,
    gaussian_nll,
    mog_nll,
)
from iclr.baselines import (
    dmmse_batch,
    ridge_batch,
)
from iclr.predictive_montecarlo import montecarlo_task_estimate
