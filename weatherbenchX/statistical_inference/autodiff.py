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
"""Autodiff of the values_from_mean_statistics function, using jax."""

from collections.abc import Hashable, Mapping
import functools
from typing import Any

import jax
import jax.numpy as jnp
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import utils
import xarray as xr


StatsValues = Mapping[str, Mapping[Hashable, xr.DataArray]]
MetricValues = Mapping[str, Mapping[Hashable, xr.DataArray]]


def per_unit_values_linearized_around_mean_statistics(
    metrics: Mapping[str, metrics_base.Metric],
    aggregation_state: aggregation.AggregationState,
    experimental_unit_dim: str,
    ) -> tuple[MetricValues, MetricValues]:
  """Per-unit metric values using linearized values_from_mean_statistics.

  To unpack this:

  We have values of our statistics for each experimental unit (which for weather
  will typically be a forecast initialization time) and the metric is a function
  of the mean of these statistics over all the experimental units.

  When this function (`values_from_mean_statistics` abbreviated here as `f`) is
  the identity, or is linear, the result of the metric is a simple mean of
  per-unit values, and many classical statistical inference procedures can be
  applied, including ones which handle temporal dependence between the units.

  We would like to extend these methods from the simple case of a mean, to
  handle more complicated metrics which may be a *non-linear* function f of the
  means of one *or more* statistics x_i.

  One approximate way to do this, is to reduce it to the easier linear case by
  linearizing f at the mean values of the statistics (i.e. replacing it with a
  first-order Taylor expansion centered around these mean values). Then we can
  apply the linearized function `f_linearized` to the per-unit values of the
  statistics, resulting in linearized per-unit values of the metric, which have
  some convenient properties:

  (1) The mean of these linearized per-unit values is the correct value of the
    metric:
      mean(f_linearized(x_i)) = f_linearized(mean(x_i)) = f(mean(x_i)).
    where the final equality holds because f_linearized is expanded around the
    mean so has no error there.
  (2) The variance of the mean of the linearized per-unit values is a reasonable
    approximation of the variance of the metric:
      Var[f(mean(x_i))]
       = (grad f)^T Cov[mean(x_i)] (grad f) + o(Tr[Cov[mean(x_i)]])
       = Var[f_linearized(mean(x_i))] + o(Tr[Cov[mean(x_i)]])
       = Var[mean(f_linearized(x_i))] + o(Tr[Cov[mean(x_i)]])

    where the o(Tr[Cov[mean(x_i)]]) term just means that the error goes away
    asymptotically as the variance (or trace of covariance in the multivariate
    case) of the mean statistics goes to zero with increasing sample size.

    This approximation is good provided that the function f is sufficiently
    linear over the range of sampling variation we expect to see for the mean
    of the statistics mean(x_i).

  Properties (1) and (2) make these linearized values good proxies to use with
  statistical inference methods designed to infer a mean of a univariate time
  series, if the real goal is to infer a smooth function `f` of the mean
  statistics.

  This code essentially supports applications of the (multivariate) Delta
  method, a very standard statistical method for approximating Var[f(X)] by
  estimating the covariance of X and then computing (grad f)^T Cov(X) (grad f).
  Our approach described above is equivalent mathematically to applying this
  method to the mean of the statistics, but it is more convenient in practise
  because it allows us to avoid estimating the covariance matrix of the mean
  statistics, which might be large if they are high-dimensional. It also allows
  us to re-use code written for univariate time-series analysis.

  A technical note about weighted means:

  To be precise, when we want a *weighted* mean of per-unit values (via the
  per-unit sum_weighted_statistics / sum_weights of the aggregation state with
  non-constant weights), the weight normalization makes this a non-linear
  function even if the `values_from_mean_statistics` itself is linear, since we
  must divide by the summed weights. We handle this by including the weight
  normalization as part of the nonlinear function f, which is differentiated as
  a function of the means of both the per-unit sum_weighted_statistics and the
  per-unit sum_weights.

  Args:
    metrics: The metrics for which you want to compute linearized per-unit
      values.
    aggregation_state: The per-unit statistics to use to compute the metric
      values.
    experimental_unit_dim: The dimension corresponding to the experimental
      units.

  Returns:
    value: The value(s) of the metrics, computed from statistics whose mean has
      been taken over all experimental units.
    per_unit_tangents: The per-unit tangent values which, when added to the
      `value` above, give linearized per-unit values of the metrics. These are
      a linear function of the difference between per-unit statistics and their
      mean values, and so they have mean zero.
      We return them separately because a typical use case is to estimate a
      variance or autocovariance and in these cases the mean would be subtracted
      anyway, so best not to compromise numerical precision by adding the
      `value` if we expect to subtract it again later.
  """
  per_unit_agg_state = aggregation_state  # Distinguish from summed over units.
  del aggregation_state

  # TODO(matthjw): Use the xarray_jax library to eliminate much of the
  # boilerplate here. For now we manually convert DataArrays to and from
  # jax.Arrays in order to interface with jax and its autodiff functionality.

  cpu_device = jax.local_devices(backend='cpu')[0]

  def _to_jax_array(data_array: xr.DataArray) -> jax.Array:
    array = data_array.data
    if isinstance(array, jax.Array):
      return array
    else:
      # JAX will place arrays on an accelerator by default, which is not what we
      # want here, since this will generally be a cheap computation on data held
      # on CPU. So we explicitly place on CPU.
      return jnp.array(array, device=cpu_device)

  def _from_jax_array(jax_array: jax.Array,
                      template_data_array: xr.DataArray,
                      extra_trailing_dims: tuple[str, ...] = (),
                      extra_coords: Mapping[str, Any] | None = None,
                      ) -> xr.DataArray:
    return xr.DataArray(
        jax_array,
        dims=template_data_array.dims + extra_trailing_dims,
        coords=template_data_array.coords | (extra_coords or {}),
    )

  experimental_unit_coord = utils.get_and_check_experimental_unit_coord(
      per_unit_agg_state, experimental_unit_dim)

  # Put the experimental unit dimension last, so we can vmap over the last
  # axis from jax later.
  per_unit_agg_state = per_unit_agg_state.map(
      lambda x: x.transpose(..., experimental_unit_dim))

  # Normally we sum the aggregation states rather than take the mean. The two
  # are ultimately equivalent since the 1/N will cancel out between the
  # numerator (sum_weighted_statistics) and denominator (sum_weights). But we
  # take the mean here because we want a Taylor expansion around the mean for
  # the numerator and denominator as separate arguments.
  mean_agg_state = per_unit_agg_state.map(
      lambda x: x.mean(experimental_unit_dim, skipna=False))
  mean_weighted_stats_jax = xarray_tree.map_structure(
      _to_jax_array, mean_agg_state.sum_weighted_statistics)
  mean_weighted_stats_template = mean_agg_state.sum_weighted_statistics
  mean_weights_jax = xarray_tree.map_structure(
      _to_jax_array, mean_agg_state.sum_weights)
  mean_weights_template = mean_agg_state.sum_weights

  result_template = None

  def metric_jax(weighted_stats_jax, weights_jax):
    weighted_stats = xarray_tree.map_structure(
        _from_jax_array, weighted_stats_jax, mean_weighted_stats_template)
    weights = xarray_tree.map_structure(
        _from_jax_array, weights_jax, mean_weights_template)
    agg_state = aggregation.AggregationState(weighted_stats, weights)
    mean_stats = agg_state.mean_statistics()
    result = metrics_base.compute_metrics_from_statistics(metrics, mean_stats)

    nonlocal result_template
    result_template = result
    return xarray_tree.map_structure(_to_jax_array, result)

  # This gives us a linearized version of the function at the mean values, which
  # maps input tangents to output tangents.
  values_jax, linearized_metric_jax = jax.linearize(
      metric_jax, mean_weighted_stats_jax, mean_weights_jax)
  # It only maps one input at a time however, which must be of the same shape as
  # the mean values passed above. So we vmap over the last axis (which we have
  # arranged to be the experimental unit dimension) in order to compute
  # linearized values for all the per-unit values in a single call.
  linearized_metric_jax = jax.vmap(
      linearized_metric_jax, in_axes=-1, out_axes=-1)

  value = xarray_tree.map_structure(
      _from_jax_array, values_jax, result_template)

  tangents_in = aggregation.AggregationState.map_multi(
      lambda x, y: x-y, per_unit_agg_state, mean_agg_state)

  tangents_weighted_stats_jax = xarray_tree.map_structure(
      _to_jax_array, tangents_in.sum_weighted_statistics)
  tangents_weights_jax = xarray_tree.map_structure(
      _to_jax_array, tangents_in.sum_weights)
  tangents_out_jax = linearized_metric_jax(
      tangents_weighted_stats_jax, tangents_weights_jax)

  tangents_out = xarray_tree.map_structure(
      functools.partial(
          _from_jax_array,
          # The results will have an extra trailing experimental unit dimension
          # relative to the shape of results in result_template, which are the
          # values of the metric at the mean values, not at all the per-unit
          # values. So we specify the extra dimension and its coord here.
          extra_trailing_dims=(experimental_unit_dim,),
          extra_coords={experimental_unit_dim: experimental_unit_coord}),
      tangents_out_jax, result_template)

  # Convert underlying arrays from jax.Array back to numpy.
  (value, tangents_out) = xarray_tree.map_structure(
      lambda x: x.as_numpy(), (value, tangents_out))

  return value, tangents_out
