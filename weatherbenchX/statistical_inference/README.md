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

* The standard t-test.
* The t-test with AR(2) autocorrelation correction from [^1]
* Both the above with delta-method confidence intervals for metrics which are
  nonlinear functions of the mean statistics.
* The IID bootstrap.
* A cluster bootstrap [^3] [^4] which assumes independence only between
  clusters.
* The stationary bootstrap of [^5], with optimal block length selection from
  [^6] [^7] but generalized to support non-linear functions of means of
  multivariate statistics via a delta-method trick.

## Methods to consider implementing in future

* Diebold-Mariano test [^8] or another of the family of tests based on HAC
  (Heteroskedasticity and Autocorrelation Consistent) variance estimators, as a
  a better-studied and more standard alternative to [^1]. These methods also
  have problems at small sample sizes and/or high degrees of autocorrelation
  however, and there are a number of choices e.g. of kernel and window length
  selection method with different trade-offs. [^10] offers a modern review and
  some practical recommendations.
* One of the second-order-correct block bootstrap CI methods of [^9] for smooth
  functions of vector means, either the studentized or the BCa-style intervals.
  Perhaps adapted to the circular block bootstrap to avoid the need to correct
  for endpoint bias. Unlike a naive application of studentized (bootstrap-t) or
  BCa intervals to the block bootstrap, these methods are more principled and
  retain the good asymptotics of studentized and BCa intervals from the IID
  case. It remains to be seen how effective they are at practical sample sizes
  though despite the improved asymptotic order, and they also add some
  complexity and further choices.

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

[^8]: Diebold, F. X. & Mariano, R. S. Comparing predictive accuracy. J. Bus. Econ.
  Stat. 20, 134–144 (2002).

[^9]: Götze, F. & Künsch, H. R. Second-order correctness of the blockwise
  bootstrap for stationary observations. Ann. Stat. 24, 1914-1933 (1996).

[^10]: Lazarus, E., Lewis, D. J., Stock, J. H. & Watson, M. W. HAR inference:
  Recommendations for practice. J. Bus. Econ. Stat. 36, 541–559 (2018).





