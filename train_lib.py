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

"""Training library."""

import os
from absl import logging
import torch
from torch import nn
from torch import optim
from torchvision import models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _initialize():
  net = models.resnet18(weights=None, num_classes=10)
  net.to(DEVICE)
  return net


def _save(net, path):
  state = net.state_dict()
  torch.save(state, path)
  logging.info('Saved a checkpoint at %s', path)


def train_or_reload_model(
    train_loader,
    val_loader,
    path,
    class_weights,
    num_epochs,
    do_saving=True,
    min_save_epoch=0,
    quiet_train=True,
):
  """Train or reload the original model."""
  net = _initialize()

  if os.path.exists(path):
    logging.info('Reloading model from %s...', path)
    net.load_state_dict(torch.load(path))
  else:
    train(
        net,
        class_weights,
        num_epochs,
        train_loader,
        eval_loader=val_loader,
        do_saving=do_saving,
        save_path=path,
        min_save_epoch=min_save_epoch,
        quiet=quiet_train,
    )
  net.eval()
  return net


def train(
    net,
    class_weights,
    num_epochs,
    train_loader,
    eval_loader=None,
    do_saving=True,
    save_path='',
    min_save_epoch=0,
    quiet=False,
):
  """Train."""
  criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
  optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-3)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=num_epochs
  )
  best_acc = 0  # best test accuracy
  logging.info('Will train for %d epochs.', num_epochs)

  for epoch in range(0, num_epochs):
    _train(
        epoch, net, train_loader, optimizer, criterion, scheduler, quiet=quiet
    )
    if eval_loader:
      if not quiet:
        logging.info('best_acc = %.3f', best_acc)
      best_acc = _test(
          epoch,
          net,
          eval_loader,
          best_acc,
          criterion,
          do_saving=do_saving if epoch >= min_save_epoch else False,
          save_path=save_path,
          quiet=quiet,
      )


def _train(epoch, net, loader, optimizer, criterion, scheduler, quiet):
  """Train helper."""
  if not quiet:
    logging.info('\nEpoch: %d', epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  # Per-class correct and total:
  list_of_classes = list(range(10))
  correct_pc = [0 for _ in list_of_classes]
  total_pc = [0 for _ in list_of_classes]

  for sample in loader:
    inputs = sample['image']
    targets = sample['age_group']
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    for c in list_of_classes:
      num_class_c = (targets == c).sum().item()
      correct_class_c = (
          ((predicted == targets) * (targets == c)).float().sum().item()
      )
      total_pc[c] += num_class_c
      correct_pc[c] += correct_class_c

  scheduler.step()
  if not quiet:
    logging.info(
        'Train acc %.3f%% (%d/%d)', 100.0 * correct / total, correct, total
    )
    for c in list_of_classes:
      logging.info(
          'Train accuracy of class %d: %.3f%% (%d/%d)',
          c,
          100.0 * correct_pc[c] / max(total_pc[c], 0.00001),
          correct_pc[c],
          total_pc[c],
      )


def _test(
    epoch,
    net,
    loader,
    best_acc,
    criterion,
    do_saving=False,
    save_path='',
    quiet=False,
):
  """Test."""
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  # Per-class correct and total:
  list_of_classes = list(range(10))
  correct_pc = [0 for c in list_of_classes]
  total_pc = [0 for c in list_of_classes]
  with torch.no_grad():
    for sample in loader:
      inputs = sample['image']
      targets = sample['age_group']
      inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

      outputs = net(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      for c in list_of_classes:
        num_class_c = (targets == c).sum().item()
        correct_class_c = (
            ((predicted == targets) * (targets == c)).float().sum().item()
        )
        total_pc[c] += num_class_c
        correct_pc[c] += correct_class_c
  if not quiet:
    logging.info(
        'Held-out acc %.3f%% (%d/%d)', 100.0 * correct / total, correct, total
    )
    for c in list_of_classes:
      logging.info(
          'Held-out accuracy of class %d: %.3f%% (%d/%d)',
          c,
          100.0 * correct_pc[c] / max(total_pc[c], 0.00001),
          correct_pc[c],
          total_pc[c],
      )

  # Save checkpoint.
  acc = 100.0 * correct / total
  if acc > best_acc:
    if do_saving:
      logging.info('New best acc %.3f%% at epoch %d', acc, epoch)
      _save(net, save_path)
    best_acc = acc
  return best_acc
