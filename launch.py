"""Spawn work on the deployed chessformer-label app.

Deployed apps outlive the local client, so a spawned call keeps running even if
this process exits or the connection drops. An ephemeral `modal run --detach`
app can stop before a spawned function is picked up.

    modal deploy label.py
    python launch.py boards train
    python launch.py label train
    python launch.py status train
"""

import sys

import modal

import label


def fn(name):
    return modal.Function.from_name("chessformer-label", name)


def main():
    stage, split = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train"
    if stage == "boards":
        per_game = (
            label.TRAIN_BOARDS_PER_GAME if split == "train" else label.TEST_BOARDS_PER_GAME
        )
        games = label.TRAIN_GAMES if split == "train" else label.TEST_GAMES
        call = fn("build_boards").spawn(split, games, per_game)
        print(f"spawned build_boards({split}, {games}, {per_game}) -> {call.object_id}")
    elif stage == "label":
        calls = [fn("label_shard").spawn(split, i) for i in range(label.NUM_SHARDS)]
        print(f"spawned {len(calls)} label shards for {split}")
    elif stage in ("pack", "train", "eval"):
        name = {"pack": "pack", "train": "train", "eval": "evaluate"}[stage]
        f = modal.Function.from_name("chessformer-train", name)
        call = f.spawn(split) if stage == "pack" else f.spawn()
        print(f"spawned {name} -> {call.object_id}")
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
