# EDA Insights

## 1. Purpose

[`1_eda_contact_tracking_video_context.ipynb`](../notebooks/1_eda_contact_tracking_video_context.ipynb)
builds the evidence base for the NFL contact detection workflow. It borrows the
starter notebook's field visualization, helmet/video synchronization, and
tracking-distance baseline, while keeping the implementation aligned with this
repo's notebook standards.

## 2. Notebook Flow

| Step | Purpose |
| --- | --- |
| Setup and configuration | Define paths, plotting defaults, target names, and fast-run flags. |
| Load data | Read labels, sample submission, tracking, helmets, and video metadata. |
| Data quality | Check schemas, missingness, duplicates, contact ID parsing, and memory use. |
| Label analysis | Measure contact balance by player-player vs player-ground contact. |
| Tracking context | Inspect player positions, speed, acceleration, orientation, and play length. |
| Helmet and video context | Check box coverage, views, frame counts, and metadata timing. |
| Field visualization | Plot an example play/step on a football field. |
| Distance baseline | Tune player-player distance thresholds with MCC. |

## 3. Initial Hypotheses

Player-player contact should be strongly related to tracking distance, but the
relationship will be noisy because player body contact is not the same as
helmet-center distance. The distance baseline should be treated as a sanity
check and lower bound, not as the final modeling path.

Player-ground contact needs a separate strategy. Since the starter distance
baseline leaves ground rows as non-contact, the next model should add features
from motion, posture proxies, helmet boxes, and temporal context.

## 4. Validation Implications

Rows from the same `game_play` are temporally and spatially correlated. EDA and
modeling notebooks should use play-grouped validation whenever they report MCC,
so a model is tested on plays it did not see during threshold tuning or model
fitting.

## 5. First Experiment

The first experiment is a cleaned-up version of the sample notebook baseline:

- parse `contact_id`;
- merge tracking coordinates for both players;
- compute Euclidean distance in yards;
- tune a hard distance threshold on validation labels;
- write `submission.csv` with the required schema.

Expected limitation: all ground rows are predicted as `0`, so recall on ground
contact will be poor until a dedicated ground-contact branch is added.
