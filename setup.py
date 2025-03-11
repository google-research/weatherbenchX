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
# ==============================================================================
import setuptools

base_requires = [
    "apache_beam[gcp]>=2.31.0",
    "cftime>=1.6.2",
    "numpy>=2.1.3",
    "pandas>=2.2.3",
    "scipy",
    "scikit-learn",
    "xarray>=2024.11.0",
    "zarr",
    "fsspec",
    "gcsfs",
    "absl-py",
    "pyarrow>=14",
]
docs_requires = [
    "myst-nb",
    "myst-parser",
    "sphinx",
    "sphinx_rtd_theme",
]
tests_requires = [
    "pytest",
    "pyink",
]
setuptools.setup(
    name="weatherbenchX",
    version="2025.03.1",
    license="Apache 2.0",
    author="Google LLC",
    author_email="weatherbenchX@google.com",
    install_requires=base_requires,
    extras_require={
        "tests": tests_requires,
        "docs": docs_requires,
    },
    url="https://github.com/google-research/weatherbenchX",
    packages=setuptools.find_packages(),
    python_requires=">=3"
)
