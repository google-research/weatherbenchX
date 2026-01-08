# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementations of categorical metrics."""

from typing import Hashable, Mapping, Sequence, Union, final

import numpy as np
from weatherbenchX.metrics import base
from weatherbenchX.metrics import wrappers
import xarray as xr
import xarray.ufuncs as xu


class TruePositives(base.PerVariableStatistic):
  """True positives from binary predictions and targets."""

  @property
  def unique_name(self) -> str:
    return 'TruePositives'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:

    return (
        (predictions.astype(bool) * targets.astype(bool))
        .where(~xu.isnan(predictions * targets))
        .astype(np.float32)
    )


class TrueNegatives(base.PerVariableStatistic):
  """True negatives from binary predictions and targets."""

  @property
  def unique_name(self) -> str:
    return 'TrueNegatives'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:

    return (
        (~predictions.astype(bool) * ~targets.astype(bool))
        .where(~xu.isnan(predictions * targets))
        .astype(np.float32)
    )


class FalsePositives(base.PerVariableStatistic):
  """False positives from binary predictions and targets."""

  @property
  def unique_name(self) -> str:
    return 'FalsePositives'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:

    return (
        (predictions.astype(bool) * ~targets.astype(bool))
        .where(~xu.isnan(predictions * targets))
        .astype(np.float32)
    )


class FalseNegatives(base.PerVariableStatistic):
  """False negatives from binary predictions and targets."""

  @property
  def unique_name(self) -> str:
    return 'FalseNegatives'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    return (
        (~predictions.astype(bool) * targets.astype(bool))
        .where(~xu.isnan(predictions * targets))
        .astype(np.float32)
    )


class SEEPS(base.Statistic):
  """Computes Stable Equitable Error in Probability Space.

  Definition in Rodwell et al. (2010):
  https://www.ecmwf.int/en/elibrary/76205-new-equitable-score-suitable-verifying-precipitation-nwp

  Important: In most cases, the statistic will contain NaNs because of the
  masking of high and low p1 values. For this reason, a `mask` coordinate will
  be added to the resulting statistic to be used in combination with
  `masked=True` in the aggregator. If a mask already exists in either the
  predictions or targets, it will be combined with the p1 mask.
  """

  def __init__(
      self,
      variables: Sequence[str],
      climatology: xr.Dataset,
      dry_threshold_mm: Union[float, Sequence[float]] = 0.25,
      min_p1: Union[float, Sequence[float]] = 0.1,
      max_p1: Union[float, Sequence[float]] = 0.85,
  ):
    # pyformat: disable
    """Init.

    Args:
      variables: List of precipitation variables to compute SEEPS for.
      climatology: Climatology containing `*_seeps_dry_fraction` and
        `*_seeps_threshold` for each of the precipitation variables with
        dimensions `dayofyear` and `hour`, as well as `latitude` and `longitude`
        corresponding to the predictions/targets coordinates, see example below.
      dry_threshold_mm: Values smaller or equal are considered dry. Unit: mm.
        Can be list for each variable. Must be same length. Default: 0.25
      min_p1: Mask out p1 values below this threshold. Can be list for each
        variable. Default: 0.1
      max_p1: Mask out p1 values above this threshold. Can be list for each
        variable. Default: 0.85

    Example:
        >>> climatology
        <xarray.Dataset> Size: 24MB
        Dimensions:                                     (hour: 4, dayofyear: 366,
                                                        longitude: 64, latitude: 32)
        Coordinates:
          * dayofyear                                   (dayofyear) int64 3kB 1 ... 366
          * hour                                        (hour) int64 32B 0 6 12 18
          * latitude                                    (latitude) float64 256B -87.1...
          * longitude                                   (longitude) float64 512B 0.0 ...
        Data variables:
            total_precipitation_6hr_seeps_dry_fraction  (hour, dayofyear, longitude, latitude) ...
            total_precipitation_6hr_seeps_threshold     (hour, dayofyear, longitude, latitude) ...
    """
    # pyformat: enable
    self._variables = variables
    self._climatology = climatology
    self._dry_threshold_mm = (
        dry_threshold_mm
        if isinstance(dry_threshold_mm, Sequence)
        else [dry_threshold_mm] * len(variables)
    )
    self._min_p1 = (
        min_p1 if isinstance(min_p1, Sequence) else [min_p1] * len(variables)
    )
    self._max_p1 = (
        max_p1 if isinstance(max_p1, Sequence) else [max_p1] * len(variables)
    )
    assert (
        len(self._variables)
        == len(self._dry_threshold_mm)
        == len(self._min_p1)
        == len(self._max_p1)
    ), 'All arguments must have the same length.'

  @property
  def unique_name(self) -> str:
    suffix = (
        '_'.join(self._variables)
        + '_dry_threshold_mm_'
        + '_'.join([str(s) for s in self._dry_threshold_mm])
        + '_min_p1_'
        + '_'.join([str(s) for s in self._min_p1])
        + '_max_p1_'
        + '_'.join([str(s) for s in self._max_p1])
    )
    return f'SEEPS_{suffix}'

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Maps computation over all variables listed in self._variables."""
    out = {}
    for variable, dry_threshold_mm, min_p1, max_p1 in zip(
        self._variables, self._dry_threshold_mm, self._min_p1, self._max_p1
    ):
      out[variable] = self._compute_seeps_per_variable(
          predictions[variable],
          targets[variable],
          variable,
          dry_threshold_mm,
          min_p1,
          max_p1,
      )
    return out

  def _convert_precip_to_seeps_cat(
      self,
      da: xr.DataArray,
      wet_threshold_for_valid_time: xr.DataArray,
      dry_threshold_mm: float,
  ):
    """Helper function for SEEPS computation. Converts values to categories."""
    # Convert to SI units [meters]
    dry_threshold = dry_threshold_mm / 1000.0
    dry = da <= dry_threshold
    light = xu.logical_and(
        da > dry_threshold, da < wet_threshold_for_valid_time
    )
    heavy = da >= wet_threshold_for_valid_time
    result = xr.concat(
        [dry, light, heavy],
        dim=xr.DataArray(['dry', 'light', 'heavy'], dims=['seeps_cat']),
    )
    # Convert NaNs back to NaNs. .where() will convert to float type.
    # Note that in the WB2 implementation, there was an additional
    # .astype('int') before the .where(). It seems to work fine without it
    # though.
    result = result.where(da.notnull())
    return result

  def _compute_seeps_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      variable: str,
      dry_threshold_mm: float,
      min_p1: float,
      max_p1: float,
  ) -> xr.DataArray:
    valid_time = predictions.init_time + predictions.lead_time
    wet_threshold = self._climatology[f'{variable}_seeps_threshold']
    wet_threshold_for_valid_time = wet_threshold.sel(
        dayofyear=valid_time.dt.dayofyear, hour=valid_time.dt.hour
    ).load()

    predictions_cat = self._convert_precip_to_seeps_cat(
        predictions, wet_threshold_for_valid_time, dry_threshold_mm
    )
    targets_cat = self._convert_precip_to_seeps_cat(
        targets, wet_threshold_for_valid_time, dry_threshold_mm
    )

    # Compute contingency table
    out = (
        predictions_cat.rename({'seeps_cat': 'forecast_cat'})
        * targets_cat.rename({'seeps_cat': 'truth_cat'})
    ).compute()

    p1 = (
        self._climatology[f'{variable}_seeps_dry_fraction']
        .mean(('hour', 'dayofyear'))
        .compute()
    )

    # Compute scoring matrix
    # The contingency table and p1 should have matching spatial dimensions.
    scoring_matrix = [
        [xr.zeros_like(p1), 1 / (1 - p1), 4 / (1 - p1)],
        [1 / p1, xr.zeros_like(p1), 3 / (1 - p1)],
        [
            1 / p1 + 3 / (2 + p1),
            3 / (2 + p1),
            xr.zeros_like(p1),
        ],
    ]
    das = []
    for mat in scoring_matrix:
      das.append(xr.concat(mat, dim=out.truth_cat))
    scoring_matrix = 0.5 * xr.concat(das, dim=out.forecast_cat)
    scoring_matrix = scoring_matrix.compute()

    # Take dot product
    result = xr.dot(out, scoring_matrix, dims=('forecast_cat', 'truth_cat'))

    # Mask out p1 thresholds
    mask = (p1 >= min_p1) & (p1 <= max_p1)
    result = result.where(mask, np.nan)

    # Add NaN mask. If mask coordinate already exists, combine them.
    if hasattr(predictions, 'mask'):
      if hasattr(targets, 'mask'):
        raise ValueError(
            'Both predictions and targets have masks. This should not happen.'
        )
      mask = mask & predictions.mask
    elif hasattr(targets, 'mask'):
      mask = mask & targets.mask

    result.coords['mask'] = mask

    return result


class RankedProbabilityScore(base.PerVariableStatistic):
  """Ranked probability score for cumulative distribution functions.

  Given a ground truth scalar random variable Y, a prediction random variable X,
  a sequence of bin boundaries b_0 < b_1 < ... < b_k, where b_0 = -inf and
  b_K = +inf, the Ranked Probability Score is defined as

    RPS = E[ Σk (CDF(Y)(b_k) - CDF(X)(b_k))^2 ]

  where the sum over k is taken over k = 1, 2, ..., K, and CDF(X) and CDF(Y)
  are the cumulative distribution functions of X and Y, respectively.

  Here it is assumed that the predictions and targets already represent the CDF
  in the `bin_dim` dimension.

  For an implementation that computes the RPS from ensemble predictions, see
  `probabilistic.EnsembleRankedProbabilityScore`.
  """

  def __init__(
      self,
      bin_dim: str,
  ):
    self._bin_dim = bin_dim

  @property
  def unique_name(self) -> str:
    return 'RankedProbabilityScore'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    return ((predictions - targets) ** 2).sum(self._bin_dim)


# Metrics
class CSI(base.PerVariableMetric):
  """Critical Success Index.

  Also called Threat Score (TS).

  CSI = (TP / (TP + FP + FN)).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['TruePositives'] / (
        statistic_values['TruePositives']
        + statistic_values['FalsePositives']
        + statistic_values['FalseNegatives']
    )


class Accuracy(base.PerVariableMetric):
  """Accuracy.

  ACC = (TP + TN) / (TP + FP + FN + TN).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
        'TrueNegatives': TrueNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return (
        statistic_values['TruePositives'] + statistic_values['TrueNegatives']
    ) / (
        statistic_values['TruePositives']
        + statistic_values['FalsePositives']
        + statistic_values['FalseNegatives']
        + statistic_values['TrueNegatives']
    )


class Recall(base.PerVariableMetric):
  """Also called True Positive Rate (TPR) or Sensitivity.

  Recall = TP / (TP + FN).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalseNegatives': FalseNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['TruePositives'] / (
        statistic_values['TruePositives'] + statistic_values['FalseNegatives']
    )


class FalseAlarmRate(base.PerVariableMetric):
  """False Alarm Rate.

  FAR = FP / (TP + FP).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['FalsePositives'] / (
        statistic_values['TruePositives'] + statistic_values['FalsePositives']
    )


class Precision(base.PerVariableMetric):
  """Also called Positive Predictive Value (PPV).

  Precision = TP / (TP + FP).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['TruePositives'] / (
        statistic_values['TruePositives'] + statistic_values['FalsePositives']
    )


class F1Score(base.PerVariableMetric):
  """F1 score.

  F1 = 2 * Precision * Recall / (Precision + Recall)
     = 2 * TP / (2 * TP + FP + FN).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return (
        2
        * statistic_values['TruePositives']
        / (
            2 * statistic_values['TruePositives']
            + statistic_values['FalsePositives']
            + statistic_values['FalseNegatives']
        )
    )


class FrequencyBias(base.PerVariableMetric):
  """Frequency bias.

  FB = PP / P = (TP + FP) / (TP + FN)
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return (
        statistic_values['TruePositives'] + statistic_values['FalsePositives']
    ) / (statistic_values['TruePositives'] + statistic_values['FalseNegatives'])


class HSS(base.PerVariableMetric):
  """Heidke Skill Score.

  HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
        'TrueNegatives': TrueNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    tp = statistic_values['TruePositives']
    tn = statistic_values['TrueNegatives']
    fp = statistic_values['FalsePositives']
    fn = statistic_values['FalseNegatives']
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return numerator / denominator


class ETS(base.PerVariableMetric):
  """Equitable Threat Score (also called Gilbert Skill Score).

  ETS = (TP - TP_random) / (TP + FP + FN - TP_random)
  where TP_random = ((TP + FP) * (TP + FN)) / (TP + FP + FN + TN).
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
        'TrueNegatives': TrueNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    tp = statistic_values['TruePositives']
    tn = statistic_values['TrueNegatives']
    fp = statistic_values['FalsePositives']
    fn = statistic_values['FalseNegatives']
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    all_sum = tp + fp + fn + tn
    tp_random = (tp_plus_fp * tp_plus_fn) / all_sum
    numerator = tp - tp_random
    denominator = tp + fp + fn - tp_random
    return numerator / denominator


class SEDI(base.PerVariableMetric):
  """Symmetric extremal dependency index.

  SEDI = (ln(F) - ln(H) + ln(1-H) - ln(1-F)) / (ln(H) + ln(F) + ln(1-H) +
  ln(1-F))
  where H = TP/(TP+FN) (hit rate) and F = FP/(FP+TN) (false alarm rate).
  See Ferro and Stephenson (2011)
  https://journals.ametsoc.org/view/journals/wefo/26/5/waf-d-10-05030_1.pdf
  """

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'TruePositives': TruePositives(),
        'FalsePositives': FalsePositives(),
        'FalseNegatives': FalseNegatives(),
        'TrueNegatives': TrueNegatives(),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    tp = statistic_values['TruePositives']
    tn = statistic_values['TrueNegatives']
    fp = statistic_values['FalsePositives']
    fn = statistic_values['FalseNegatives']

    h = tp / (tp + fn)
    f = fp / (fp + tn)

    # Clip rates to avoid log(0) errors and division by zero, following
    # Ferro and Stephenson (2011)
    h = h.clip(1e-6, 1 - 1e-6)
    f = f.clip(1e-6, 1 - 1e-6)

    log_h = np.log(h)
    log_f = np.log(f)
    log_1_minus_h = np.log(1 - h)
    log_1_minus_f = np.log(1 - f)

    numerator = log_f - log_h + log_1_minus_h - log_1_minus_f
    denominator = log_h + log_f + log_1_minus_h + log_1_minus_f

    return numerator / denominator


class Reliability(base.PerVariableMetric):
  """Reliability / calibration curve.

  E.g. see
  https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html.

  This metric should be used for binary ground truth and probability
  predictions. It will automatically apply binning to the predictions into 10
  equal-width bins, assuming the predictions are in [0, 1]. You can modify the
  number of bins and the bin edges by passing in `bin_values` and `bin_dim`. For
  each bin of predicted probabilities, the metric will compute the probability
  of the positive class according to the ground truth.
  """

  def __init__(
      self,
      bin_values: Sequence[float] = (
          -np.inf,
          0.1,
          0.2,
          0.3,
          0.4,
          0.5,
          0.6,
          0.7,
          0.8,
          0.9,
          1.
      ),
      bin_dim: str = 'reliability_bin',
      statistic_suffix: str | None = None,
  ):
    self._bin_values = bin_values
    self._bin_dim = bin_dim
    self._unique_name_suffix = statistic_suffix

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    binned_prediction_wrapper = wrappers.ContinuousToBins(
        which='predictions',
        bin_values=self._bin_values,
        bin_dim=self._bin_dim,
        unique_name_suffix=self._unique_name_suffix,
    )
    return {
        'TruePositives': wrappers.WrappedStatistic(
            TruePositives(), binned_prediction_wrapper
        ),
        'FalsePositives': wrappers.WrappedStatistic(
            FalsePositives(), binned_prediction_wrapper
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['TruePositives'] / (
        statistic_values['TruePositives'] + statistic_values['FalsePositives']
    )


class Confident(base.PerVariableStatisticWithClimatology):
  """Forecast confidence.

  Whether the prediction spread < threshold * climatological spread.
  """

  def __init__(
      self,
      ensemble_dim: str,
      climatology: xr.Dataset,
      spread_quantile_boundaries: tuple[float, float] = (0.1, 0.9),
      confidence_threshold: float = 0.7,
  ):
    super().__init__(climatology)
    self._ensemble_dim = ensemble_dim
    self._spread_low, self._spread_high = spread_quantile_boundaries
    self._confidence_threshold = confidence_threshold

  @property
  def unique_name(self) -> str:
    return (
        'Confident'
        + f'_conf_thres={self._confidence_threshold}'
        + f'_spread_low={self._spread_low}'
        + f'_spread_high={self._spread_high}'
    )

  def _compute_per_variable_with_aligned_climatology(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      aligned_climatology: xr.DataArray,
  ) -> xr.DataArray:
    """Computes confidence per variable."""
    del targets  # Unused.

    # Get the spread of the predictions.
    predictions_spread = predictions.quantile(
        self._spread_high, dim=self._ensemble_dim
    ) - predictions.quantile(self._spread_low, dim=self._ensemble_dim)

    # Climatologies are already quantiles.
    climatology_spread = aligned_climatology.sel(
        quantile=self._spread_high
    ) - aligned_climatology.sel(quantile=self._spread_low)

    return predictions_spread < self._confidence_threshold * climatology_spread


class Covered(base.PerVariableStatistic):
  """Forecast coverage.

  Whether the target lies within a prediction interval with specified quantile
  boundaries.
  """

  def __init__(
      self,
      ensemble_dim: str,
      interval_quantile_boundaries: tuple[float, float] = (0.1, 0.9),
  ):
    self._ensemble_dim = ensemble_dim
    self._interval_low, self._interval_high = interval_quantile_boundaries

  @property
  def unique_name(self) -> str:
    return (
        'Covered'
        + f'_interval_low={self._interval_low}'
        + f'_interval_high={self._interval_high}'
    )

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    """Computes coverage per variable."""
    predictions_low = predictions.quantile(
        self._interval_low, dim=self._ensemble_dim
    )
    predictions_high = predictions.quantile(
        self._interval_high, dim=self._ensemble_dim
    )
    return (predictions_low <= targets) & (targets <= predictions_high)


class JaccardDistant(base.PerVariableStatisticWithClimatology):
  """Thresholded Jaccard distance of prediction interval from climatology.

  Whether the Jaccard distance between the forecast prediction interval and
  climatology prediction interval is greater than a threshold.

  Jaccard Distance is defined as 1 - |A ∩ B| / |A ∪ B|, where A is the set of
  points in the forecast interval and B is the set of points in the climatology
  interval.
  """

  def __init__(
      self,
      ensemble_dim: str,
      climatology: xr.Dataset,
      threshold: float = 0.75,
      interval_quantile_boundaries: tuple[float, float] = (0.1, 0.9),
  ):
    super().__init__(climatology)
    self._ensemble_dim = ensemble_dim
    self._threshold = threshold
    self._interval_low, self._interval_high = interval_quantile_boundaries

  @property
  def unique_name(self) -> str:
    return (
        'JaccardDistant'
        + f'_threshold={self._threshold}'
        + f'_interval_low={self._interval_low}'
        + f'_interval_high={self._interval_high}'
    )

  def _compute_per_variable_with_aligned_climatology(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      aligned_climatology: xr.DataArray,
  ) -> xr.DataArray:
    """Computes jaccard distance per variable."""
    del targets  # Unused.
    predictions_low = predictions.quantile(
        self._interval_low, dim=self._ensemble_dim
    )
    predictions_high = predictions.quantile(
        self._interval_high, dim=self._ensemble_dim
    )
    climatology_low = aligned_climatology.sel(quantile=self._interval_low)
    climatology_high = aligned_climatology.sel(quantile=self._interval_high)

    # A ∩ B = max(min(A), min(B)) - min(max(A), max(B))
    max_of_lows = xu.maximum(predictions_low, climatology_low)
    min_of_highs = xu.minimum(predictions_high, climatology_high)

    # Length of intersection is the difference. If they don't overlap,
    # this difference could be negative, so we take the max with 0.
    intersection_length = xu.maximum(0, min_of_highs - max_of_lows)

    # |A ∪ B| = |A| + |B| - |A ∩ B|
    predictions_interval_length = predictions_high - predictions_low
    climatology_interval_length = climatology_high - climatology_low
    union_length = (
        predictions_interval_length + climatology_interval_length
    ) - intersection_length

    # We need to handle the case where union_length is 0. This occurs if both
    # intervals are identical single points (e.g., [5, 5] and [5, 5]). In this
    # specific case, the Jaccard Index should be 1 (perfect overlap). Note that
    # in this case the intersection_length will also be 0.
    jaccard_index = xr.where(
        union_length > 0,
        intersection_length / union_length,
        1.0
    )

    jaccard_distance = 1 - jaccard_index
    return jaccard_distance > self._threshold


class Opportunism(base.PerVariableMetric):
  """Opporunism.

  Fraction of forecast that is (un)confident, (un)covered, and
  (un)jaccard-distant.
  """

  def __init__(
      self,
      ensemble_dim: str,
      climatology: xr.Dataset,
      is_confident: bool,
      is_covered: bool | None = None,
      is_jaccard_distant: bool | None = None,
      confidence_quantile_boundaries: tuple[float, float] = (0.1, 0.9),
      coverage_quantile_boundaries: tuple[float, float] = (0.1, 0.9),
      jaccard_distance_quantile_boundaries: tuple[float, float] = (0.1, 0.9),
      confidence_threshold: float = 0.7,
      jaccard_distance_threshold: float = 0.75,
  ):
    """Initializes the Opportunism metric.

    Args:
      ensemble_dim: The dimension name of the ensemble.
      climatology: The climatology dataset.
      is_confident: Whether to compute if the forecast is confident or not in
        the metric.
      is_covered: Whether to compute if the forecast is covered or not in the
        metric. If not set, the coverage will not be computed.
      is_jaccard_distant: Whether to compute if the forecast is jaccard-distant
        or not in the metric. If not set, the jaccard-distance will not be
        computed.
      confidence_quantile_boundaries: The quantiles boundaries to use.
      coverage_quantile_boundaries: The quantiles boundaries to use.
      jaccard_distance_quantile_boundaries: The quantiles boundaries to use.
      confidence_threshold: The threshold to use for confidence.
      jaccard_distance_threshold: The threshold to use for jaccard-distance.
    """

    self._is_confident = is_confident
    self._is_covered = is_covered
    self._is_jaccard_distant = is_jaccard_distant
    self._ensemble_dim = ensemble_dim
    self._climatology = climatology
    self._confidence_quantile_boundaries = confidence_quantile_boundaries
    self._coverage_quantile_boundaries = coverage_quantile_boundaries
    self._jaccard_distance_quantile_boundaries = (
        jaccard_distance_quantile_boundaries
    )
    self._confidence_threshold = confidence_threshold
    self._jaccard_distance_threshold = jaccard_distance_threshold

  @final
  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    # Always compute confidence.
    statistics = {
        'Confident': Confident(
            ensemble_dim=self._ensemble_dim,
            climatology=self._climatology,
            spread_quantile_boundaries=self._confidence_quantile_boundaries,
            confidence_threshold=self._confidence_threshold,
        ),
    }
    # Conditionally compute coverage and jaccard-distance if they're actually
    # being used.
    if self._is_covered is not None:
      statistics['Covered'] = Covered(
          ensemble_dim=self._ensemble_dim,
          interval_quantile_boundaries=self._coverage_quantile_boundaries,
      )
    if self._is_jaccard_distant is not None:
      statistics['JaccardDistant'] = JaccardDistant(
          ensemble_dim=self._ensemble_dim,
          climatology=self._climatology,
          threshold=self._jaccard_distance_threshold,
          interval_quantile_boundaries=self._jaccard_distance_quantile_boundaries,
      )
    return statistics

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes opportunism per variable."""
    confident = statistic_values['Confident']
    if self._is_confident:
      statistics_values = confident
    else:
      statistics_values = 1 - confident

    if self._is_covered is not None:
      covered = statistic_values['Covered']
      if self._is_covered:
        statistics_values = statistics_values * covered
      else:
        statistics_values = statistics_values * (1 - covered)

    if self._is_jaccard_distant is not None:
      jaccard_distant = statistic_values['JaccardDistant']
      if self._is_jaccard_distant:
        statistics_values = statistics_values * jaccard_distant
      else:
        statistics_values = statistics_values * (1 - jaccard_distant)

    return statistics_values
