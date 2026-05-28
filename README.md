# NFL Player Contact Detection

![NFL football banner](https://cdn2.unrealengine.com/the-super-bowl-concludes-the-nfl-season-but-there-s-still-plenty-of-football-left-in-madden-24-1920x1080-5a2655f237f2.jpg)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-NFL%20Player%20Contact%20Detection-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Metric](https://img.shields.io/badge/Metric-MCC-2E7D32?style=flat-square)
![Status](https://img.shields.io/badge/Status-Notebook%207%20Challenger-2E7D32?style=flat-square)

This repository contains a notebook-first workflow for Kaggle's
**1st and Future - Player Contact Detection** competition. The project detects
external contact between NFL players, and between players and the ground, using
player tracking data, baseline helmet detections, video metadata, and game
video.

## 1. Project Overview

The competition is a player-safety problem. The NFL and AWS want a reliable way
to identify contact moments throughout a football play, including body contact
with the ground. Better contact labels can support injury surveillance,
workload analysis, and future mitigation research.

This repo is organized as a Kaggle notebook workflow:

1. Understand the data through EDA and visual checks.
2. Build a submission-safe baseline.
3. Improve tabular tracking features.
4. Add temporal smoothing and contact-type-specific thresholds.
5. Investigate video and helmet-derived features as the next source of lift.

## 2. Task and Goal

For every allowable `contact_id`, predict whether contact occurred at a
specific 10 Hz timestep.

Contact types:

| Contact Type | Meaning |
| --- | --- |
| Player-player | Two NFL players are externally contacting each other. |
| Player-ground | One player's body is contacting the ground; player 2 is encoded as `G`. |

Submission format:

```text
contact_id,contact
58168_003392_0_38590_43854,0
58168_003392_0_38590_41257,1
```

The final Kaggle artifact must be named `submission.csv`.

## 3. Evaluation Metric

The competition metric is **Matthews Correlation Coefficient (MCC)**. MCC is a
better fit than accuracy because contact events are rare.

Current EDA shows:

| Target Slice | Contact Rate |
| --- | ---: |
| Overall | 1.37% |
| Ground | 4.09% |
| Player-player | 1.11% |

Project evaluation rules:

1. Validate by grouped `game_play` splits.
2. Tune thresholds on held-out plays.
3. Report ground and player-player slices separately.
4. Compare local MCC with public and private leaderboard MCC.
5. Keep the selected model conservative when public/private scores disagree.

## 4. Current Progress

Notebook 5 is the current scored champion. Notebook 7 is the active local
challenger and should be submitted next.

| Rank | Notebook | Submission | Local MCC | Public MCC | Private MCC | Status |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `5_type_specific_thresholds.ipynb` | Type-Specific Thresh, Version 3 | 0.67650 | 0.65170 | 0.65127 | Current scored champion |
| 2 | `7_blended_type_models.ipynb` | Blended type models | 0.67944 | Pending | Pending | Submit challenger |
| 3 | `6_type_specific_models.ipynb` | Type-Specific Model, Version 2 | 0.67530 | 0.65212 | 0.64025 | Rejected: private regression |
| 4 | `4_nearest_player_and_smoothing.ipynb` | Nearest Player, Version 3 | 0.67455 | 0.64497 | 0.64763 | Superseded |
| 5 | `3_tracking_feature_model.ipynb` | Tracking Feature, Version 3 | 0.65310 | 0.63075 | 0.62593 | Superseded |
| 6 | `2_distance_baseline_first_experiment.ipynb` | Distance baseline | 0.51863 | - | - | Superseded |

Progress summary:

1. Tracking-only features produced the first competitive submission.
2. Nearest-player context and temporal smoothing lifted both public and private
   scores.
3. Type-specific thresholds improved player-player precision and became the
   scored champion.
4. Fully separate type-specific models slightly improved public score but hurt
   private score, so they were rejected.
5. Blended type-specific models improved local validation and are the next
   leaderboard challenger.

## 5. Key Insights

EDA and model iterations point to several stable lessons:

1. Player-player distance is the strongest simple signal. Contact becomes rare
   beyond roughly `2` yards.
2. Ground contact behaves differently from player-player contact and needs
   dedicated motion/window features.
3. Temporal smoothing helps because contact labels and real contact moments can
   shift across neighboring 10 Hz steps.
4. Public leaderboard gains should not override private-score regression.
5. Helmet visibility and helmet-pair pixel distance show promising video-based
   signal before any heavy CNN work.

The Team Hydrogen 2nd-place writeup also suggests a longer-term direction:
temporal video crops, synchronized Sideline/Endzone views, tracking features
encoded with video context, helmet interpolation, and stage-2 blending.

## 6. Data and Kaggle Path

All notebooks use the fixed Kaggle competition path:

```text
/kaggle/input/competitions/nfl-player-contact-detection
```

Required files:

| File | Purpose |
| --- | --- |
| `train_labels.csv` | Training labels for player-player and player-ground candidates. |
| `sample_submission.csv` | Submission schema and required test rows. |
| `train_player_tracking.csv`, `test_player_tracking.csv` | 10 Hz tracking data. |
| `train_baseline_helmets.csv`, `test_baseline_helmets.csv` | Baseline helmet boxes and player assignments. |
| `train_video_metadata.csv`, `test_video_metadata.csv` | Video timing metadata for frame synchronization. |
| `train/`, `test/` | MP4 videos for each play. |

## 7. Notebook Workflow

| Notebook | Purpose |
| --- | --- |
| `1_eda_contact_tracking_video_context.ipynb` | EDA, data quality, contact balance, tracking context, helmet/video metadata, and video overlay demo. |
| `2_distance_baseline_first_experiment.ipynb` | Starter-style distance baseline with grouped validation and `submission.csv` generation. |
| `3_tracking_feature_model.ipynb` | Tracking-feature classifier for player-player and ground rows with MCC threshold tuning. |
| `4_nearest_player_and_smoothing.ipynb` | Adds nearest-player density features and play/pair probability smoothing. |
| `5_type_specific_thresholds.ipynb` | Tunes separate ground and player-player thresholds after smoothing. |
| `6_type_specific_models.ipynb` | Trains separate ground and player-player models before smoothing and threshold tuning. |
| `7_blended_type_models.ipynb` | Blends unified and type-specific probabilities before smoothing and threshold tuning. |
| `8_yolo_video_feature_probe.ipynb` | Research-mode video probe with helmet overlays, CPU YOLO install/download, and cheap video-derived features. |

Each notebook prints a `NOTEBOOK_VERSION` string at the top. Kaggle results
should only be trusted when the printed version matches
[`docs/3_model_evaluation_progress.md`](docs/3_model_evaluation_progress.md).

## 8. Repository Structure

```text
.
|-- README.md
|-- docs/
|   |-- coding_standards.md
|   |-- 1_instructions.md
|   |-- 2_eda_insights.md
|   `-- 3_model_evaluation_progress.md
`-- notebooks/
    |-- 1_eda_contact_tracking_video_context.ipynb
    |-- 2_distance_baseline_first_experiment.ipynb
    |-- 3_tracking_feature_model.ipynb
    |-- 4_nearest_player_and_smoothing.ipynb
    |-- 5_type_specific_thresholds.ipynb
    |-- 6_type_specific_models.ipynb
    |-- 7_blended_type_models.ipynb
    `-- 8_yolo_video_feature_probe.ipynb
```

## 9. Next Steps

1. Submit Notebook 7 and compare public/private MCC against Notebook 5.
2. Rerun Notebook 8 with internet enabled and confirm
   `YOLO_VIDEO_PROBE_V3_CPU`.
3. Use the video probe output to decide whether Notebook 9 should add helmet
   visibility, helmet-pair pixel distance, and interpolated box features.
4. Move toward multi-fold grouped validation before heavier video models or a
   stage-2 blend.

Final submission notebooks must run with internet disabled. YOLO can be used
for research notebooks, but any final YOLO path needs offline Kaggle inputs for
packages and weights.
