# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Launcher for running the evaluation metric with PyTorch with GPUs.

Usage:
xmanager launch launch.py

# BEGIN GOOGLE-INTERNAL
xmanager launch third_party/deepmind/unlearning_evaluation/launch.py -- \
    --xm_resource_alloc="group:xcloud/xcloud-shared-user"
# END GOOGLE-INTERNAL
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib import framework_defaults
from xmanager.contrib.internal import requirements_flag


_LOCATION = flags.DEFINE_string(
    'location',
    None,
    (
        'Where to schedule this workload (uses cell selection if unspecified) '
        '- e.g. xcloud-europe-west4'
    ),
)
_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'unlearning-metric', 'Name of the experiment.', short_name='n'
)
_ACCELERATOR = requirements_flag.DEFINE_requirements(
    'platform',
    'v100=1',
    'Accelerator specification. Format: <GPU>=<count> or <TPU>=<topology>.',
    short_name='t',
)
_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    '/gcs/unlearning-gdm-xcloud-bucket/',
    'Directory containing the data.',
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '/gcs/unlearning-gdm-xcloud-bucket/checkpoints/',
    'Directory containing the checkpoints.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/gcs/unlearning-gdm-xcloud-bucket/outputs/',
    'Directory to write the outputs.',
)


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with xm_abc.create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    job_requirements = xm.JobRequirements(
        **_ACCELERATOR.value, location=_LOCATION.value
    )
    executor = xm_abc.executors.Gcp(requirements=job_requirements)

    executable_args = {
        'data_dir': _DATA_DIR.value,
        'checkpoint_dir': _CHECKPOINT_DIR.value,
        'output_dir': _OUTPUT_DIR.value,
    }
    (executable,) = experiment.package([
        xm.python_container(
            # Package the current directory that this script is in.
            path='.',
            base_image=framework_defaults.base_image(
                'us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.1-12.py310',
                job_requirements.accelerator,
            ),
            entrypoint=xm.ModuleName('unlearning_evaluation.main'),
            use_deep_module=True,
            executor_spec=executor.Spec(),
            args=executable_args,
        )
    ])
    job = xm.Job(executable, executor)
    experiment.add(job)


if __name__ == '__main__':
  app.run(main)
