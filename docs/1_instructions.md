# Competition Instructions and Project Approach

## 1. Objective

Kaggle's **NFL Player Contact Detection** competition asks competitors to
predict contact events during football plays. Each `contact_id` represents a
single `game_play`, 10 Hz `step`, `player1`, and either `player2` or the ground
marker `G`.

The target is binary:

- `contact = 1`: the player is externally contacting another player or ground.
- `contact = 0`: no contact for that pair at that moment.

## 2. Input Files

Expected Kaggle path:

```text
/kaggle/input/nfl-player-contact-detection
```

Core files:

- `train_labels.csv`: training contact labels.
- `sample_submission.csv`: required submission rows and schema.
- `train_player_tracking.csv`: training tracking data.
- `test_player_tracking.csv`: test tracking data.
- `train_baseline_helmets.csv`: training baseline helmet detections.
- `test_baseline_helmets.csv`: test baseline helmet detections.
- `train_video_metadata.csv`: timing metadata for training videos.

Video files are provided under the competition `train/` and `test/` folders and
are useful for visual inspection and later computer-vision features.

## 3. Evaluation

Submissions are evaluated with Matthews Correlation Coefficient between
predicted and actual contact labels. MCC rewards balanced binary classification
and is more informative than accuracy for this imbalanced contact task.

The submission file must be named:

```text
submission.csv
```

and use this schema:

```text
contact_id,contact
58168_003392_0_38590_43854,0
58168_003392_0_38590_41257,1
```

## 4. Code Competition Requirements

This is a Kaggle Code Competition. Final submissions must run as Kaggle
notebooks with internet disabled and complete within the competition runtime
limit. Publicly available external data and pretrained models are allowed when
attached through Kaggle-compatible inputs.

Notebook defaults should therefore be offline-safe and avoid runtime package
installation.

## 5. Project Plan

The repository starts with two notebooks:

1. Run EDA in
   [`1_eda_contact_tracking_video_context.ipynb`](../notebooks/1_eda_contact_tracking_video_context.ipynb)
   to understand labels, tracking, helmets, metadata, and baseline failure
   modes.
2. Build the first submission-safe experiment in
   [`2_distance_baseline_first_experiment.ipynb`](../notebooks/2_distance_baseline_first_experiment.ipynb)
   using the starter notebook's tracking-distance idea with cleaner validation.

Recommended next experiments:

1. Add tracking-derived relative speed, acceleration, orientation, and nearest
   player features.
2. Build a ground-contact branch because the distance baseline cannot identify
   player-ground contact.
3. Add temporal smoothing by play and pair after out-of-fold validation.
4. Add helmet/video features once the tracking-only baseline is stable.
