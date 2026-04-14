import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import imageio
import imageio.v3 as iio
import stable_retro as retro
from tqdm.auto import trange


@dataclass
class Args:
    out_root: str = "data/retro_experiment"
    episodes_per_game: int = 3
    steps_per_episode: int = 500
    max_games: int | None = None
    games: str | None = None
    seed: int = 42
    image_ext: str = "jpg"
    save_mp4: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect (frame_t, frame_t+1, action_label, env_id) samples from Stable-Retro"
    )
    parser.add_argument("--out-root", type=str, default=Args.out_root)
    parser.add_argument("--episodes-per-game", type=int, default=Args.episodes_per_game)
    parser.add_argument("--steps-per-episode", type=int, default=Args.steps_per_episode)
    parser.add_argument("--max-games", type=int, default=Args.max_games)
    parser.add_argument(
        "--games",
        type=str,
        default=Args.games,
        help="Comma-separated game IDs (e.g. 1942-Nes-v0,AddamsFamily-Sms-v0)",
    )
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--image-ext", choices=["jpg", "png"], default=Args.image_ext)
    parser.add_argument("--save-mp4", action="store_true", default=Args.save_mp4)
    return parser.parse_args()


def has_rom(game_name: str) -> bool:
    try:
        retro.data.get_romfile_path(game_name)
        return True
    except FileNotFoundError:
        return False


def get_games(options: argparse.Namespace) -> list[str]:
    games_with_roms = [game for game in retro.data.list_games() if has_rom(game)]

    if options.games:
        requested = [name.strip() for name in options.games.split(",") if name.strip()]
        requested_set = set(requested)
        filtered = [game for game in games_with_roms if game in requested_set]
        missing = sorted(requested_set.difference(filtered))
        if missing:
            print("Skipping unavailable/unimported games:", ", ".join(missing))
        games_with_roms = filtered

    if options.max_games is not None:
        games_with_roms = games_with_roms[: options.max_games]

    return games_with_roms


def action_to_label(action: list[int], buttons: list[str]) -> tuple[str, list[str]]:
    """Convert an action vector to a human-readable label and list of pressed buttons."""
    pressed = []
    for index, value in enumerate(action):
        if int(value) != 1:
            continue
        button_name = buttons[index]
        if button_name is None:
            continue
        pressed.append(button_name)
    label = "NOOP" if not pressed else "+".join(sorted(pressed))
    return label, pressed


def sample_clean_action(buttons: list[str]) -> list[int]:
    """Sample a cleaner, human-like action: <=1 direction + <=1 button."""
    name_to_index = {
        name: index
        for index, name in enumerate(buttons)
        if isinstance(name, str)
    }

    directions = [name for name in ["LEFT", "RIGHT", "UP", "DOWN"] if name in name_to_index]

    non_direction_buttons = [
        name
        for name in name_to_index
        if name not in {"LEFT", "RIGHT", "UP", "DOWN"}
    ]

    chosen_direction = random.choice([None] + directions)
    chosen_button = random.choice([None] + non_direction_buttons)

    action = [0] * len(buttons)
    if chosen_direction is not None:
        action[name_to_index[chosen_direction]] = 1
    if chosen_button is not None:
        action[name_to_index[chosen_button]] = 1
    return action


def save_frame(image, frame_path: Path) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(frame_path, image)


def save_episode_video(frames: list, output_path: Path, fps: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path.as_posix(), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main() -> None:
    options = parse_args()
    random.seed(options.seed)

    out_root = Path(options.out_root)
    frames_root = out_root / "frames"
    metadata_path = out_root / "transitions.jsonl"
    summary_path = out_root / "summary.json"
    out_root.mkdir(parents=True, exist_ok=True)

    games = get_games(options)
    print(f"Found {len(games)} games to sample")
    if not games:
        print("No games available. Import ROMs first.")
        return

    total_transitions = 0
    per_game_counts: dict[str, int] = {}
    action_counter: Counter[str] = Counter()

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for game in games:
            try:
                env = retro.make(
                    game=game,
                    render_mode="rgb_array",
                    use_restricted_actions=retro.Actions.ALL,
                )
            except Exception as error:
                print(f"Skipping {game}: cannot create env ({error})")
                continue

            buttons = list(env.buttons)
            game_transition_count = 0

            for episode_index in trange(
                options.episodes_per_game,
                desc=f"Sampling {game}",
            ):
                obs, _ = env.reset(seed=options.seed + episode_index)
                episode_frames = [obs] if options.save_mp4 else None

                for step_index in range(options.steps_per_episode):
                    action = sample_clean_action(buttons)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    if episode_frames is not None:
                        episode_frames.append(next_obs)

                    action_label, pressed_buttons = action_to_label(action, buttons)
                    action_counter[action_label] += 1

                    frame_t = (
                        frames_root
                        / game
                        / f"ep_{episode_index:04d}"
                        / f"step_{step_index:05d}_t.{options.image_ext}"
                    )
                    frame_tp1 = (
                        frames_root
                        / game
                        / f"ep_{episode_index:04d}"
                        / f"step_{step_index:05d}_tp1.{options.image_ext}"
                    )

                    save_frame(obs, frame_t)
                    save_frame(next_obs, frame_tp1)

                    record = {
                        "game": game,
                        "system": game.split("-")[-2] if "-v" in game else game.split("-")[-1],
                        "episode": episode_index,
                        "step": step_index,
                        "frame_t": str(frame_t.as_posix()),
                        "frame_tp1": str(frame_tp1.as_posix()),
                        "action_vector": action,
                        "action_label": action_label,
                        "pressed_buttons": pressed_buttons,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                    metadata_file.write(json.dumps(record) + "\n")

                    game_transition_count += 1
                    total_transitions += 1

                    obs = next_obs
                    if terminated or truncated:
                        obs, _ = env.reset()
                        if episode_frames is not None:
                            episode_frames.append(obs)

                if episode_frames is not None:
                    video_path = (
                        out_root
                        / "videos"
                        / game
                        / f"ep_{episode_index:04d}.mp4"
                    )
                    save_episode_video(episode_frames, video_path)

            env.close()
            per_game_counts[game] = game_transition_count
            print(f"Collected {game_transition_count} transitions for {game}")

    summary = {
        "out_root": str(out_root.as_posix()),
        "metadata": str(metadata_path.as_posix()),
        "total_games": len(per_game_counts),
        "total_transitions": total_transitions,
        "per_game_transitions": per_game_counts,
        "top_action_labels": action_counter.most_common(30),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done")
    print(f"Transitions: {total_transitions}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
