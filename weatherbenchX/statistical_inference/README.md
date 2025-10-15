# Statistical inference for evaluation metrics in weatherbenchX

Evaluation metrics are computed on a finite sample of predictions and targets,
but we usually think of them as estimates of a true underlying metric which
would be attained using an infinite population drawn from the same distribution.
Alongside a point estimate of this true underlying metric, we would also like to
have:

* Confidence intervals
* Estimated standard errors (standard deviation of the estimate under resampling
  of the dataset)

And for differences in evaluation metrics between two models, we would like:

* Significance tests (including p-values) for the null hypothesis that the
  true underlying metric is the same for the two models (difference equals zero)
  against a typically two-sided alternative.
* Confidence intervals and standard errors for the difference.

Because of the general framework in which metrics are defined in weatherbenchX,
it is possible to implement a number of different methods for computing the
above in a way that can be applied to a broad class of metrics. These methods do
rest on statistical assumptions however, which will not hold for all metrics,
for all sources of data on which they are computed, or for all sample sizes. So
care must be taken in their application and interpretation.

While there is no perfect method for some of the challenging cases which weather
metrics present, we aim to curate a selection of sensible default methods with
relatively broad applicability, whose assumptions and caveats are documented.
We welcome input from the statistical community on how to improve the selection
of defaults offered.

## Handling dependence

Most standard methods for producing confidence intervals assume that
experimental units are independent. This is not usually an appropriate
assumption for weather metrics due to temporal dependence between statistics at
nearby forecast initialization times. We implement some statistical methods here
which are capable of handling this temporal dependence, however they do come
with further caveats and trade-offs.

Spatial correlation can be an issue too, although we typically avoid the need
to account for this by treating spatial locations (on a grid, for example) as
fixed, and averaging over them before performing any statistical inference.

## Methods implemented

* The standard t-test, for metrics which are simple means (including linear
  functions of means.)
* The t-test with AR(2) autocorrelation correction from [^1], again for metrics
  which are means.
* The IID bootstrap, with BCa intervals [^2] as well as percentile intervals
* A cluster bootstrap [^3] [^4] which assumes independence only between
  clusters, also with BCa intervals.
* The stationary bootstrap of [^5], with optimal block length selection [^6]
  [^7]. BCa intervals are supported with the help of the stationary jackknife
  [^8].

## Methods to consider implementing in future

* Diebold-Mariano test [^9] using the Newey-West HAC (Heteroskedasticity and
  Autocorrelation Consistent) variance estimator as a more principled
  alternative to [^1] for metrics which are means.

The above methods require no knowledge of the gradient of the
`values_from_mean_statistics` function. If we implement this gradient for our
Metrics (either manually or via autodiff) a range of further options open up:

* A multivariate t-test for the mean statistics, followed by use of the gradient
  for a 1st-order approximation of the effect on variance of the
  `values_from_mean_statistics` transformation.
* As above for the Diebold-Mariano / HAC test.
* Studentized ('bootstrap-t') bootstrap intervals [^10], using the same approach
  as above (HAC variance estimator for the mean statistics together with
  gradient of `values_from_mean_statistics`) to avoid the need for an expensive
  inner bootstrap to estimate per-bootstrap-replicate standard errors.
* One of the second-order-correct blockwise bootstrap methods of [^11] for
  smooth functions of vector means. Perhaps adapted to the stationary bootstrap
  to avoid the need to correct for endpoint bias.
* For all block bootstrap methods: automated block length selection applied to
  the linearization of `values_from_mean_statistics` around the mean statistics,
  computed separately at each timestep. The mean of these would be equal to the
  result, meaning the use of a block length selection method designed for mean
  estimates would be better motivated, and preferable to applying these methods
  to the per-timestep values of `values_from_mean_statistics` itself whose mean
  may be very different from the end result in non-linear cases, and whose
  non-linearities applied on a per-timestep basis can destroy some relevant
  autocorrelation.

[^1]: A. J. Geer, Significance of changes in medium-range forecast scores.
  Tellus A Dyn. Meterol. Oceanogr. 68, 30229 (2016).

[^2]: Efron, B. Better bootstrap confidence intervals. J.A.S.A. 82, 171-185
  (1987)

[^3]: Davison, A. C. & Hinkley, D. V. Bootstrap Methods and their Application
  (Cambridge University Press, 1997), pp.100-101.

[^4]: Sherman, M. & le Cessie, Saskia, A comparison between bootstrap methods
  and generalized estimating equations for correlated outcomes in generalized
  linear models, Communications in Statistics - Simulation and Computation,
  26:3, 901-925 (1997).

[^5]: Politis, D. N. & Romano, J. P. The stationary bootstrap. J.A.S.A. 89,
  1303-1313 (1994).

[^6]: Politis, D. N. & White, H. Automatic Block-Length Selection for the
  Dependent Bootstrap, Econometric Reviews, 23:1, 53-70 (2004).

[^7]: Patton, A., Politis, D. N. & White, H. Correction to "Automatic
  Block-Length Selection for the Dependent Bootstrap" by D. Politis and
  H. White, Econometric Reviews, 28:4, 372-375 (2009).

[^8]: Zhou, W. & Lahiri, S. Stationary jackknife. J. Time Ser. Anal. 45,
  333-360 (2024).

[^9]: Diebold, F. X. & Mariano, R. S. Comparing predictive accuracy. J. Bus. Econ.
  Stat. 20, 134–144 (2002).

[^10]: Efron, B. & Tibshirani, R. J. An Introduction to the Bootstrap. (Chapman &
  Hall, 1993), p.160.

[^11]: Götze, F. & Künsch, H. R. Second-order correctness of the blockwise
  bootstrap for stationary observations. Ann. Stat. 24, 1914-1933 (1996).

