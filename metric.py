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

"""Evaluation metric code."""

import numpy as np
import torch


# =============================================================================
# Metric code.
DELTA = 1e-05
EPSILON_CAP = 50
NUM_THRESHOLDS_PER_UNIT = 100


def compute_logit_scaled_confidence(logits, targets):
  """Transform logits into logit-scaled confidences."""

  # The below code is a numerically-stable approach for computing the
  # logit-scaled confidences, which is adapted from the LiRA codebase
  # (see this link: https://github.com/tensorflow/privacy/blob/
  # 45da453410ffa078b2d05dc4883d006d578e1b6d/research/mi_lira_2021/
  # score.py#LL38C9-L54C83)

  count = logits.shape[0]

  # Remove the largest logit, for numerical reasons.
  logits -= np.max(logits, axis=1, keepdims=True)

  # Apply the softmax.
  probs = np.array(np.exp(logits), dtype=np.float64)
  probs /= np.sum(probs, axis=1, keepdims=True)

  # For each example, get the confidence of the correct class.
  prob_correct = probs[np.arange(count), targets[:count]]

  # Now, set the correct positions to 0s, to sum only the incorrect ones.
  probs[np.arange(count), targets[:count]] = 0
  prob_wrong = np.sum(probs, axis=1)

  # Each of prob_correct and prob_wrong have shape [batch_size] and take values
  # between 0 and 1. conf will also have shape [batch_size] but takes values
  # outside that range too.
  conf = np.log(prob_correct + 1e-45) - np.log(prob_wrong + 1e-45)

  # prob_correct is the "confidence" before the logit-scaling (returning it only
  # for plotting reasons).
  return prob_correct, conf


def _get_double_threshold_rates(pos_confs_, neg_confs_):
  """return tpr, fnr, fpr, tnr using a double-threshold decision rule family."""
  # This rule uses two thresholds and predicts that any confidences in
  # between them belong to the peakier distribution.
  pos_diff = pos_confs_.max() - pos_confs_.min()
  neg_diff = neg_confs_.max() - neg_confs_.min()
  smallest = neg_diff if pos_diff > neg_diff else pos_diff

  # For simplicity, define 'positive' as the peakiest distirbution.
  if pos_diff >= neg_diff:
    pos_confs_, neg_confs_ = neg_confs_, pos_confs_

  # Get the right thresholds.
  buffer = 2.0
  width = smallest
  min_ = (pos_confs_.min() + width - buffer).detach().cpu().numpy()
  max_ = (pos_confs_.max() + buffer).detach().cpu().numpy()
  num_right_thresholds = int(np.ceil((max_ - min_) * NUM_THRESHOLDS_PER_UNIT))
  right_thresholds_ = np.linspace(min_, max_, num=num_right_thresholds)
  width = width.detach().cpu().numpy()

  # Get the left thresholds.
  num_left_thresholds = int(np.ceil(2 * buffer * NUM_THRESHOLDS_PER_UNIT))
  thr_left_ = np.concatenate(
      [
          np.linspace(
              tr - width - buffer, tr - width + buffer, num=num_left_thresholds
          )
          for tr in right_thresholds_
      ],
      axis=0,
  )
  thr_right_ = np.tile(
      np.reshape(right_thresholds_, (-1, 1)), (1, num_left_thresholds)
  ).reshape((-1))
  thresholds_flat = [(l, r) for l, r in zip(thr_left_, thr_right_)]
  total_num_thresholds = len(thresholds_flat)

  # thr_right_, thr_left_ are [num_thresholds] and {pos/neg}_confs are
  # [num models] make them all into [num models, num_thresholds].
  num_models = pos_confs_.shape[0]
  thr_right_ = np.tile(np.expand_dims(thr_right_, 0), (num_models, 1))
  thr_left_ = np.tile(np.expand_dims(thr_left_, 0), (num_models, 1))
  pos_confs_ = torch.tile(pos_confs_[:, None], (1, total_num_thresholds))
  neg_confs_ = torch.tile(neg_confs_[:, None], (1, total_num_thresholds))

  thr_right_ = torch.from_numpy(thr_right_).cuda()
  thr_left_ = torch.from_numpy(thr_left_).cuda()

  # Predicted positives (pp) / predicted negatives (pn):
  def _is_pp(c):
    return torch.logical_and(
        torch.le(thr_left_, c), torch.le(c, thr_right_)
    ).type(torch.float32)

  def _is_pn(c):
    return torch.logical_not(_is_pp(c)).type(torch.float32)

  # The mean is over the models dimension.
  # [num_thresholds].
  tpr = torch.mean(_is_pp(pos_confs_), 0)
  fnr = torch.mean(_is_pn(pos_confs_), 0)
  fpr = torch.mean(_is_pp(neg_confs_), 0)
  tnr = torch.mean(_is_pn(neg_confs_), 0)

  return tpr, fnr, fpr, tnr, thresholds_flat


def _get_single_threshold_rates(pos_confs_, neg_confs_):
  """Get tpr, fnr, fpr, tnr using a single-threshold decision rule family."""
  # This rule uses one threshold and predicts that any confidences to
  # the right belong to the class whose median is largest.

  all_confs = torch.cat([pos_confs_, neg_confs_])
  min_val = all_confs.min()
  max_val = all_confs.max()

  num_thresholds = int(
      torch.ceil((max_val - min_val) * NUM_THRESHOLDS_PER_UNIT)
  )
  thresholds_ = torch.linspace(min_val, max_val, num_thresholds)
  thresholds_flat = list(thresholds_.detach().cpu().numpy())
  thresholds_ = thresholds_.cuda()

  # thresholds_ is [num_thresholds] and pos_confs/neg_confs is [num models]
  # make them all into [num models, num_thresholds]
  num_models = pos_confs_.shape[0]
  thresholds_ = torch.tile(thresholds_[None, :], (num_models, 1))
  pos_confs_ = torch.tile(pos_confs_[:, None], (1, num_thresholds))
  neg_confs_ = torch.tile(neg_confs_[:, None], (1, num_thresholds))

  # The mean is over the models dimension.
  # [num_thresholds]. For each threshold, the true positive rate.
  tpr = torch.mean((pos_confs_ >= thresholds_).type(torch.float32), 0)
  fpr = torch.mean((neg_confs_ >= thresholds_).type(torch.float32), 0)
  tnr = torch.mean((neg_confs_ < thresholds_).type(torch.float32), 0)
  fnr = torch.mean((pos_confs_ < thresholds_).type(torch.float32), 0)

  return tpr, fnr, fpr, tnr, thresholds_flat


def _get_epsilons(pos_confs, neg_confs, delta):
  """Get epsilons."""
  epsilons = []

  # Collect and return auxiliary info useful for debugging or visualizations.
  all_fprs, all_tprs, all_fnrs, all_tnrs, thresholds, best_thresh_idx = (
      [],
      [],
      [],
      [],
      [],
      [],
  )

  num_examples = pos_confs.shape[1]
  for i in range(num_examples):
    pos_confs_ = pos_confs[:, i].reshape(-1)
    neg_confs_ = neg_confs[:, i].reshape(-1)

    pos_diff = np.max(pos_confs_) - np.min(pos_confs_)
    neg_diff = np.max(neg_confs_) - np.min(neg_confs_)
    largest = pos_diff if pos_diff > neg_diff else neg_diff
    smallest = neg_diff if pos_diff > neg_diff else pos_diff
    if smallest / largest < 0.01:
      # If this ratio is below 1% means that the unlearned distribution is very
      # peaky. In that case, return the maximum epsilon, because epsilon should
      # be really bad. The metric computation might not be sensitive enough to
      # identify this in all cases, so catch that case here to be sure.
      epsilons.append(EPSILON_CAP)
      all_fprs.append(-1)
      all_tprs.append(-1)
      all_fnrs.append(-1)
      all_tnrs.append(-1)
      thresholds.append(-1)
      best_thresh_idx.append(-1)
      continue

    pos_confs_ = torch.from_numpy(pos_confs_).cuda()
    neg_confs_ = torch.from_numpy(neg_confs_).cuda()

    # Compute the tpr / fnr / fpr / tnr using two different decision rules.
    tpr_d, fnr_d, fpr_d, tnr_d, tf_d = _get_double_threshold_rates(
        pos_confs_, neg_confs_
    )
    tpr_s, fnr_s, fpr_s, tnr_s, tf_s = _get_single_threshold_rates(
        pos_confs_, neg_confs_
    )
    tpr = torch.cat([tpr_d, tpr_s], 0)
    fnr = torch.cat([fnr_d, fnr_s], 0)
    fpr = torch.cat([fpr_d, fpr_s], 0)
    tnr = torch.cat([tnr_d, tnr_s], 0)

    # thresholds_flat = np.array(tf_d + tf_s)
    thresholds_flat = tf_d + tf_s

    total_num_thresholds = tpr.shape[0]
    thr_eps = torch.zeros(total_num_thresholds).cuda()

    # Handle the special case of perfect separation (fpr = fnr = 0).
    # If the sum fpr + fnr is zero, it means both are 0 (since both are >=0).
    fpr_fnr_both_zero = torch.eq(
        fpr + fnr, torch.zeros(total_num_thresholds).cuda()
    )
    tpr_tnr_both_ones = torch.eq(
        tpr + tnr, 2 * torch.ones(total_num_thresholds).cuda()
    )
    thr_eps = torch.where(
        fpr_fnr_both_zero * tpr_tnr_both_ones,
        torch.full(thr_eps.shape, np.inf).cuda(),
        thr_eps,
    )

    # Discard thresholds where either fpr or fnr is 0 for a threshold (but not
    # both; handled above). The multiplication with `thr_eps_is_zero` will
    # exclude cases where both fpr = fnr = 0 because the code in the above lines
    # would have set those entires of `thr_eps` to inf. This is based on the
    # hypothesis that that's an indication of insufficient samples.
    thr_eps_is_zero = torch.eq(thr_eps, torch.zeros_like(thr_eps).cuda())
    fpr_or_fnr_is_zero = torch.eq(fpr * fnr, torch.zeros_like(thr_eps).cuda())
    thr_eps = torch.where(
        thr_eps_is_zero * fpr_or_fnr_is_zero,
        torch.full(thr_eps.shape, np.nan).cuda(),
        thr_eps,
    )

    # For the surviving thresholds (epsilon not nan nor inf), compute epsilon:
    thr_eps = torch.where(
        torch.eq(thr_eps, torch.zeros_like(thr_eps).cuda()),
        torch.from_numpy(
            np.clip(
                np.nanmax(
                    [
                        np.log(1 - delta - fpr.detach().cpu().numpy())
                        - np.log(fnr.detach().cpu().numpy()),
                        np.log(1 - delta - fnr.detach().cpu().numpy())
                        - np.log(fpr.detach().cpu().numpy()),
                    ],
                    axis=0,
                ),
                0,
                None,
            )
        ).cuda(),
        thr_eps,
    )

    # Find the inds where thr_eps is not nan
    keep_inds = [
        ind
        for ind, te in enumerate(thr_eps.detach().cpu().numpy())
        if not np.isnan(te)
    ]
    thr_eps = thr_eps.detach().cpu().numpy()[keep_inds]
    tpr = tpr.detach().cpu().numpy()[keep_inds]
    fpr = fpr.detach().cpu().numpy()[keep_inds]
    tnr = tnr.detach().cpu().numpy()[keep_inds]
    fnr = fnr.detach().cpu().numpy()[keep_inds]
    # kept_thresholds = thresholds_flat[keep_inds]
    kept_thresholds = [thresholds_flat[ind] for ind in keep_inds]

    assert thr_eps, "Something went wrong, all thresholds gave nan epsilon..."
    thr_eps = np.array(thr_eps)
    epsilons.append(np.nanmax(thr_eps))
    # Get the best threshold via something like 'arg-nan-max': set nans to zeros
    # and find the max of the rest of the entries.
    best_thresh = np.argmax(
        np.where(np.isnan(thr_eps), np.zeros_like(thr_eps), thr_eps)
    )
    all_fprs.append(np.array(fpr))
    all_tprs.append(np.array(tpr))
    all_fnrs.append(np.array(fnr))
    all_tnrs.append(np.array(tnr))
    thresholds.append(kept_thresholds)
    best_thresh_idx.append(best_thresh)

  epsilons = np.clip(epsilons, 0, EPSILON_CAP)
  return epsilons


def compute_forget_score_from_confs(unlearned_confs, retrained_confs):
  """Returns the forget score based on the unlearned and retrained confidences.

  Args:
    unlearned_confs: [num models, num examples]. The confidence of each forget
      set example under each of the unlearned models.
    retrained_confs: [num models, num examples]. The confidence of each forget
      set example under each of the retrained models.
  """
  _, forget_set_size = retrained_confs.shape
  retrain_means = np.median(retrained_confs, axis=0)  # [num examples].
  unlearned_means = np.median(unlearned_confs, axis=0)  # [num examples].

  # The positive class *for each example* is defined as whichever of the two
  # classes (retrained vs unlearned) has a larger median.
  retrain_is_positive = retrain_means > unlearned_means  # [num examples].
  pos_confs = np.where(retrain_is_positive, retrained_confs, unlearned_confs)
  neg_confs = np.where(retrain_is_positive, unlearned_confs, retrained_confs)
  eps = _get_epsilons(pos_confs, neg_confs, delta=DELTA)

  # Back-of-the-envelope calculation for the max value of epsilon
  # based on the given number of samples from each distribution.
  num_models = unlearned_confs.shape[0]
  max_epsilon = np.ceil(np.log(num_models - 1))

  # Compute the score.
  start_idx = 0
  bucket_size = 0.5
  end_idx = start_idx + bucket_size
  points = 1
  bucket_points = {}
  while end_idx <= max_epsilon:
    bucket_points[start_idx] = points
    points /= 2
    start_idx += bucket_size
    end_idx = start_idx + bucket_size

  forget_score = 0
  for e in eps:
    for start_idx, points in bucket_points.items():
      if e < start_idx + bucket_size:
        forget_score += points
        break

  forget_score /= forget_set_size
  return forget_score
