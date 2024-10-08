import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    # Create train, test from train split. Ensure train, test come from different game plays
    old_train = pd.read_csv(raw / "train_labels.csv")
    unique_game_play = old_train["game_play"].unique()
    new_train_game_play, new_test_game_play = train_test_split(
        unique_game_play, test_size=0.1, random_state=0
    )

    new_train = old_train[old_train["game_play"].isin(new_train_game_play)]
    new_test = old_train[old_train["game_play"].isin(new_test_game_play)]
    assert set(new_train["contact_id"]).isdisjoint(
        set(new_test["contact_id"])
    ), "Train and test label share samples!"

    new_train.to_csv(public / "train_labels.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # baseline helmets
    old_train_baseline_helmets = pd.read_csv(raw / "train_baseline_helmets.csv")
    new_train_baseline_helmets = old_train_baseline_helmets[
        old_train_baseline_helmets["game_play"].isin(new_train_game_play)
    ]
    new_test_baseline_helmets = old_train_baseline_helmets[
        old_train_baseline_helmets["game_play"].isin(new_test_game_play)
    ]

    new_train_baseline_helmets.to_csv(public / "train_baseline_helmets.csv", index=False)
    new_test_baseline_helmets.to_csv(public / "test_baseline_helmets.csv", index=False)

    # player tracking
    old_train_player_tracking = pd.read_csv(raw / "train_player_tracking.csv")
    new_train_player_trackings = old_train_player_tracking[
        old_train_player_tracking["game_play"].isin(new_train_game_play)
    ]
    new_test_player_trackings = old_train_player_tracking[
        old_train_player_tracking["game_play"].isin(new_test_game_play)
    ]

    new_train_player_trackings.to_csv(public / "train_player_tracking.csv", index=False)
    new_test_player_trackings.to_csv(public / "test_player_tracking.csv", index=False)

    # video metadata
    old_train_video_metadata = pd.read_csv(raw / "train_video_metadata.csv")
    new_train_video_metadata = old_train_video_metadata[
        old_train_video_metadata["game_play"].isin(new_train_game_play)
    ]
    new_test_video_metadata = old_train_video_metadata[
        old_train_video_metadata["game_play"].isin(new_test_game_play)
    ]

    new_train_video_metadata.to_csv(public / "train_video_metadata.csv", index=False)
    new_test_video_metadata.to_csv(public / "test_video_metadata.csv", index=False)

    # Copy over videos
    for game_play_type in ["All29", "Endzone", "Sideline"]:
        for game_play in new_train["game_play"].unique():
            shutil.copyfile(
                src=raw / "train" / f"{game_play}_{game_play_type}.mp4",
                dst=public / "train" / f"{game_play}_{game_play_type}.mp4",
            )

        for game_play in new_test["game_play"].unique():
            shutil.copyfile(
                src=raw / "train" / f"{game_play}_{game_play_type}.mp4",
                dst=public / "test" / f"{game_play}_{game_play_type}.mp4",
            )

    # Check integrity of the files copied
    num_train_videos_found = len(list(public.glob("train/*.mp4")))
    num_test_videos_found = len(list(public.glob("test/*.mp4")))
    num_expected_train_videos = (
        len(new_train["game_play"].unique()) * 3
    )  # *3 for All29, Endzone, Sideline
    num_expected_test_videos = len(new_test["game_play"].unique()) * 3

    assert (
        num_train_videos_found == num_expected_train_videos
    ), f"Expected {num_expected_train_videos} images, found {num_train_videos_found}"
    assert (
        num_test_videos_found == num_expected_test_videos
    ), f"Expected {num_expected_test_videos} images, found {num_test_videos_found}"

    # Create a sample submission file
    submission_df = pd.DataFrame(
        {
            "contact_id": new_test["contact_id"],
            "contact": 0,
        }
    )
    submission_df.to_csv(public / "sample_submission.csv", index=False)
