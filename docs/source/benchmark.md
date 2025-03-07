# Reproducing the WeatherBench scores

This is a guide on how to reproduce the scores and figures on the official [WeatherBench website](https://sites.research.google/weatherbench/).

## Running the evaluation script

The main evaluation script is located here: [https://github.com/google-research/weatherbench2/blob/main/public_benchmark/run_benchmark_evaluation.py](https://github.com/google-research/weatherbench2/blob/main/public_benchmark/run_benchmark_evaluation.py)

The script works in combination with a config file that defines the data loader settings for a given model/ground truth, year and resolution: [https://github.com/google-research/weatherbench2/blob/main/public_benchmark/public_configs.py](https://github.com/google-research/weatherbench2/blob/main/public_benchmark/public_configs.py)

This config file uses data on the public WeatherBench bucket, see [Data Guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).

Here is an example of how to run the script locally:
```shell
python run_benchmark_evaluation.py \
  --config=public_configs \
  --prediction=hres \
  --target=era5 \
  --resolution=64x32 \
  --year=2020 \
  --time_start=2020-01-01 \
  --time_stop=2020-01-01T12 \
  --lead_time_start=0 \
  --lead_time_stop=12 \
  --lead_time_frequency=6 \
  --output_dir=./results/ \
  --runner=DirectRunner
```

This will only work for small data. To run a full evaluation, use [Dataflow](https://weatherbench-x.readthedocs.io/en/latest/beam_dataflow.html):
```shell
export BUCKET=my-bucket
export PROJECT=my-project
export REGION=us-central1

python run_benchmark_evaluation.py \
  --config=public_configs \
  --prediction=hres \
  --target=era5 \
  --resolution=64x32 \
  --year=2020 \
  --time_start=2020-01-01 \
  --time_stop=2020-01-01T12 \
  --lead_time_start=0 \
  --lead_time_stop=12 \
  --lead_time_frequency=6 \
  --output_dir=gs://$BUCKET/tmp/ \
  --runner=DataflowRunner \
  -- \
  --project=$PROJECT \
  --region=$REGION \
  --temp_location=gs://$BUCKET/tmp/ \
  --setup_file=../setup.py \
  --job_name=wbx-evaluation
```

The precomputed results can be found here: [gs://weatherbench2/benchmark_results](https://console.cloud.google.com/storage/browser/weatherbench2/benchmark_results).

## Combining the results
For further use, e.g. to produce the scorecards or interactive graphics, we combine the results into a single file. This is done with this script: [https://github.com/google-research/weatherbench2/blob/main/public_benchmark/combine_results.py](https://github.com/google-research/weatherbench2/blob/main/public_benchmark/combine_results.py)

Deterministic and probabilistic results are processed separately. The script runs locally and can take a few minutes.

```shell
python combine_results.py \
    --input_dir=gs://weatherbench2/benchmark_results \
    --output_dir=./ \
    --mode=deterministic
# or --mode=probabilistic
```

## Plot the scorecards

To plot the scorecards, follow this notebook: [https://github.com/google-research/weatherbench2/blob/main/public_benchmark/WB_X_Website_Scorecard.ipynb](https://github.com/google-research/weatherbench2/blob/main/public_benchmark/WB_X_Website_Scorecard.ipynb)

## Interactive graphics

The code for the interactive graphics (Deterministic and Probabilistic tabs on the website) can be found here: [https://github.com/google-research/weatherbench2/blob/main/public_benchmark/apps](https://github.com/google-research/weatherbench2/blob/main/public_benchmark/apps)

See the README for a brief guide how to run the apps.