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

"""Main file to launch unlearning evaluation using the competition metric."""

import copy
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from unlearning_evaluation import metric
from unlearning_evaluation import surf
from unlearning_evaluation import train_lib

_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    'unlearning/SURF',
    'Path to the dataset.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    './checkpoints',
    'Path to the checkpoint directory.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './outputs',
    'Path to the output directory.',
)

_NUM_MODELS = flags.DEFINE_integer(
    'num_models',
    512,
    'Number of models to train.',
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Info for training the original and retrained models:
NUM_EPOCHS = 30


def do_unlearning(
    retain_loader,
    forget_loader,
    val_loader,
    class_weights,
    original_model,
    print_accuracy=False,
):
  """Run simple unlearning by finetuning."""
  del class_weights

  unlearned_model = copy.deepcopy(original_model)

  epochs = 1
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(
      unlearned_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs
  )

  unlearned_model.train()
  for _ in range(epochs):
    for sample in retain_loader:
      inputs = sample['image']
      targets = sample['age_group']
      inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

      optimizer.zero_grad()
      outputs = unlearned_model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
    scheduler.step()

    if print_accuracy:
      unlearned_model.eval()
      surf.compute_accuracy_surf(
          ['retain', 'forget', 'val'],
          [retain_loader, forget_loader, val_loader],
          unlearned_model,
          'Finetune model',
          print_=True,
      )
      unlearned_model.train()

  unlearned_model.eval()
  return unlearned_model


def _get_confs(net, loader):
  """Returns the confidences of the data in loader extracted from net."""
  confs = []
  for sample in loader:
    inputs = sample['image']
    targets = sample['age_group']
    inputs = inputs.to(DEVICE)
    logits = net(inputs)
    logits = logits.detach().cpu().numpy()
    _, conf = metric.compute_logit_scaled_confidence(logits, targets)
    confs.append(conf)
  confs = np.concatenate(confs, axis=0)
  return confs


def get_unlearned_and_retrained_confs_and_accs(
    train_loader,
    val_loader,
    test_loader,
    retain_loader,
    forget_loader,
    forget_loader_no_shuffle,
    class_weights,
    retrained_confs_path,
):
  """Returns the confidence and accuracies of unlearned and retrained models."""
  # Step 1) Get the confidences and accuracies under the unlearned models
  #######################################################################
  unlearned_confs_forget = []
  unlearned_retain_accs, unlearned_test_accs, unlearned_forget_accs = [], [], []

  # Reload the original model from which all unlearning runs will start.
  original_model = train_lib.train_or_reload_model(
      train_loader,
      val_loader,
      path=os.path.join(_CHECKPOINT_DIR.value, 'original_model.pth'),
      class_weights=class_weights,
      num_epochs=NUM_EPOCHS,
      do_saving=True,
  )

  for i in range(_NUM_MODELS.value):
    net = do_unlearning(
        retain_loader,
        forget_loader,
        val_loader,
        class_weights,
        original_model,
    )
    net.eval()

    # For this particular model, compute the forget set confidences.
    confs_forget = _get_confs(net, forget_loader_no_shuffle)
    unlearned_confs_forget.append(confs_forget)
    # For this particular model, compute the retain and test accuracies.
    accs = surf.compute_accuracy_surf(
        ['retain', 'forget', 'test'],
        [retain_loader, forget_loader, test_loader],
        net,
        'Unlearned model {}'.format(i),
        print_=False,
        print_per_class_=False,
    )
    unlearned_retain_accs.append(accs['retain'])
    unlearned_test_accs.append(accs['test'])
    unlearned_forget_accs.append(accs['forget'])

  unlearned_confs_forget = np.stack(unlearned_confs_forget)

  # Step 2) Get the confidences and accuracies under the retrained models
  #######################################################################
  recompute = True
  retrained_confs_forget = []
  retrain_retain_accs, retrain_test_accs, retrain_forget_accs = [], [], []

  if os.path.exists(retrained_confs_path):
    loaded_results = np.load(retrained_confs_path)
    # retrained_confs is [num models, num examples].
    assert loaded_results['retrained_confs'].shape[0] == _NUM_MODELS.value
    retrained_confs_forget = loaded_results['retrained_confs']
    retrain_retain_accs = loaded_results['retrain_retain_accs']
    retrain_test_accs = loaded_results['retrain_test_accs']
    retrain_forget_accs = loaded_results['retrain_forget_accs']
    recompute = False

  if recompute:
    for i in range(_NUM_MODELS.value):
      path = os.path.join(_CHECKPOINT_DIR.value, str(i))
      net = train_lib.train_or_reload_model(
          retain_loader,
          val_loader,
          path,
          class_weights,
          num_epochs=NUM_EPOCHS,
          do_saving=i < _NUM_MODELS.value,
          min_save_epoch=20,
      )
      # For this particular model, compute the forget set confidences.
      confs_forget = _get_confs(net, forget_loader_no_shuffle)
      retrained_confs_forget.append(confs_forget)
      # For this particular model, compute the retain and test accuracies.
      accs = surf.compute_accuracy_surf(
          ['retain', 'forget', 'test'],
          [retain_loader, forget_loader, test_loader],
          net,
          'Retrained model {}'.format(i),
          print_=False,
          print_per_class_=False,
      )
      retrain_retain_accs.append(accs['retain'])
      retrain_test_accs.append(accs['test'])
      retrain_forget_accs.append(accs['forget'])

    retrained_confs_forget = np.stack(retrained_confs_forget)

    np.savez(
        retrained_confs_path,
        retrained_confs=retrained_confs_forget,
        retrain_retain_accs=retrain_retain_accs,
        retrain_test_accs=retrain_test_accs,
        retrain_forget_accs=retrain_forget_accs,
    )
    logging.info('Saved retrained info to %s', retrained_confs_path)

  return (
      unlearned_confs_forget,
      retrained_confs_forget,
      unlearned_retain_accs,
      unlearned_test_accs,
      unlearned_forget_accs,
      retrain_retain_accs,
      retrain_test_accs,
      retrain_forget_accs,
  )


def main(unused_args):
  logging.info('Running on device: %s', DEVICE.upper())
  logging.info('torch version: %s', torch.__version__)

  if not os.path.isdir(_CHECKPOINT_DIR.value):
    os.mkdir(_CHECKPOINT_DIR.value)

  if not os.path.isdir(_OUTPUT_DIR.value):
    os.mkdir(_OUTPUT_DIR.value)
  retrained_confs_path = os.path.join(_OUTPUT_DIR.value, 'retrain_confs.npz')

  (
      train_loader,
      val_loader,
      test_loader,
      retain_loader,
      forget_loader,
      forget_loader_no_shuffle,
      class_weights,
  ) = surf.get_dataset(
      batch_size=64, quiet=False, dataset_path=_DATA_DIR.value
  )

  (
      unlearned_confs_forget,
      retrained_confs_forget,
      unlearned_retain_accs,
      unlearned_test_accs,
      unlearned_forget_accs,
      retrain_retain_accs,
      retrain_test_accs,
      retrain_forget_accs,
  ) = get_unlearned_and_retrained_confs_and_accs(
      train_loader,
      val_loader,
      test_loader,
      retain_loader,
      forget_loader,
      forget_loader_no_shuffle,
      class_weights,
      retrained_confs_path,
  )

  u_r_mean = np.mean(unlearned_retain_accs)
  u_t_mean = np.mean(unlearned_test_accs)
  u_f_mean = np.mean(unlearned_forget_accs)
  r_r_mean = np.mean(retrain_retain_accs)
  r_t_mean = np.mean(retrain_test_accs)
  r_f_mean = np.mean(retrain_forget_accs)

  # Also log the accuracies.
  logging.info(
      'Unlearned retain acc mean: %.2f, std %.2f.',
      u_r_mean,
      np.std(unlearned_retain_accs),
  )
  logging.info(
      'Unlearned test acc mean: %.2f, std %.2f.',
      u_t_mean,
      np.std(unlearned_test_accs),
  )
  logging.info(
      'Unlearned forget acc mean: %.2f, std %.2f.',
      u_f_mean,
      np.std(unlearned_forget_accs),
  )

  logging.info(
      'Retrain retain acc mean: %.2f, std %.2f.',
      r_r_mean,
      np.std(retrain_retain_accs),
  )
  logging.info(
      'Retrain test acc mean: %.2f, std %.2f.',
      r_t_mean,
      np.std(retrain_test_accs),
  )
  logging.info(
      'Retrain forget acc mean: %.2f, std %.2f.',
      r_f_mean,
      np.std(retrain_forget_accs),
  )

  forget_score = metric.compute_forget_score_from_confs(
      unlearned_confs_forget, retrained_confs_forget
  )
  logging.info('Forget score: %.6f', forget_score)

  final_score = forget_score * (u_r_mean / r_r_mean) * (u_t_mean / r_t_mean)
  logging.info('Final score (after utility adjustment): %.6f', final_score)


if __name__ == '__main__':
  app.run(main)
