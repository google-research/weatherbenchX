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
