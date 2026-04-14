"""Platformer-focused Stable-Retro transition sampler.

This script collects (frame_t, frame_t+1, action_label, env_id) transitions using
a progression-oriented macro-action policy:

- Chooses from weighted platformer macros (e.g., RIGHT, RIGHT+A, RIGHT+B).
- Holds each chosen macro for a random duration (hold_min..hold_max) to avoid
    frame-to-frame jitter from re-sampling actions every step.
- Detects visual stagnation via lightweight frame signatures and temporarily
    injects a recovery macro for `recovery_hold` steps.

Outputs:
- Saved frame pairs per step.
- `transitions.jsonl` with action metadata.
- `summary.json` with per-game counts and top action labels.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import imageio
import imageio.v3 as iio
import stable_retro as retro
from tqdm.auto import trange


PLATFORMER_NAME_KEYWORDS = (
    "sonic",
    "mario",
    "adventureisland",
    "ninjagaiden",
    "castlevania",
    "donkeykong",
    "kidchameleon",
    "rocketknight",
    "ristar",
    "sparkster",
    "quackshot",
    "megaman",
)


@dataclass
class Args:
    out_root: str = "data/retro_platformer_experiment"
    episodes_per_game: int = 3
    steps_per_episode: int = 500
    max_games: int | None = None
    games: str | None = None
    seed: int = 42
    image_ext: str = "jpg"
    save_mp4: bool = False
    platformer_mode: str = "tag_or_name"
    all_games: bool = False

    # Macro-action timing controls:
    # - hold_min/hold_max: sampled duration for each chosen macro action.
    # - stuck_threshold: consecutive near-identical frames before recovery fires.
    # - recovery_hold: duration of the forced recovery macro (e.g., RIGHT+A).
    hold_min: int = 4
    hold_max: int = 12
    stuck_threshold: int = 30
    recovery_hold: int = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect platformer-focused (frame_t, frame_t+1, action_label, env_id) "
            "samples from Stable-Retro"
        )
    )
    parser.add_argument("--out-root", type=str, default=Args.out_root)
    parser.add_argument("--episodes-per-game", type=int, default=Args.episodes_per_game)
    parser.add_argument("--steps-per-episode", type=int, default=Args.steps_per_episode)
    parser.add_argument("--max-games", type=int, default=Args.max_games)
    parser.add_argument(
        "--games",
        type=str,
        default=Args.games,
        help="Comma-separated game IDs (e.g. SonicTheHedgehog-Sms-v0,NinjaGaiden-Nes-v0)",
    )
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--image-ext", choices=["jpg", "png"], default=Args.image_ext)
    parser.add_argument("--save-mp4", action="store_true", default=Args.save_mp4)
    parser.add_argument(
        "--platformer-mode",
        choices=["tag", "tag_or_name"],
        default=Args.platformer_mode,
        help="How to detect platformers when --all-games is not set",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        default=Args.all_games,
        help="Disable platformer filtering and sample all selected games",
    )
    parser.add_argument("--hold-min", type=int, default=Args.hold_min)
    parser.add_argument("--hold-max", type=int, default=Args.hold_max)
    parser.add_argument("--stuck-threshold", type=int, default=Args.stuck_threshold)
    parser.add_argument("--recovery-hold", type=int, default=Args.recovery_hold)
    return parser.parse_args()


def has_rom(game_name: str) -> bool:
    try:
        retro.data.get_romfile_path(game_name)
        return True
    except FileNotFoundError:
        return False


def get_game_metadata(game_name: str) -> dict:
    metadata_path = Path(retro.data.path()) / "stable" / game_name / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def platformer_reason(game_name: str, mode: str) -> str | None:
    metadata = get_game_metadata(game_name)
    tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
    if isinstance(tags, list) and "platformer" in tags:
        return "tag"

    if mode == "tag":
        return None

    lowered = game_name.lower()
    for keyword in PLATFORMER_NAME_KEYWORDS:
        if keyword in lowered:
            return f"name:{keyword}"
    return None


def get_games(options: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    games_with_roms = [game for game in retro.data.list_games() if has_rom(game)]

    if options.games:
        requested = [name.strip() for name in options.games.split(",") if name.strip()]
        requested_set = set(requested)
        filtered = [game for game in games_with_roms if game in requested_set]
        missing = sorted(requested_set.difference(filtered))
        if missing:
            print("Skipping unavailable/unimported games:", ", ".join(missing))
        games_with_roms = filtered

    reason_by_game: dict[str, str] = {}
    if not options.all_games:
        platformer_games = []
        for game in games_with_roms:
            reason = platformer_reason(game, options.platformer_mode)
            if reason is not None:
                platformer_games.append(game)
                reason_by_game[game] = reason
        games_with_roms = platformer_games

    if options.max_games is not None:
        games_with_roms = games_with_roms[: options.max_games]

    return games_with_roms, reason_by_game


def action_to_pressed_buttons(action: list[int], buttons: list[str]) -> list[str]:
    pressed = []
    for index, value in enumerate(action):
        if int(value) != 1:
            continue
        button_name = buttons[index]
        if button_name is None:
            continue
        pressed.append(button_name)
    return pressed


def frame_signature(image) -> str:
    sampled = image[::8, ::8]
    return hashlib.blake2b(sampled.tobytes(), digest_size=8).hexdigest()


class PlatformerMacroSampler:
    # Samples temporally-correlated actions (macro actions) and holds each macro
    # for multiple emulator steps, instead of re-sampling every frame.
    def __init__(
        self,
        buttons: list[str],
        hold_min: int,
        hold_max: int,
        stuck_threshold: int,
        recovery_hold: int,
    ):
        self.buttons = buttons
        self.name_to_index = {
            name: index
            for index, name in enumerate(buttons)
            if isinstance(name, str)
        }
        self.hold_min = hold_min
        self.hold_max = hold_max
        self.stuck_threshold = stuck_threshold
        self.recovery_hold = recovery_hold

        self.remaining_hold = 0
        self.current_action = [0] * len(buttons)
        self.current_label = "NOOP"
        # Counts consecutive low-change transitions for stuck detection.
        self.stagnant_steps = 0

        self.macros = self._build_macros()

    def _build_action(self, button_names: list[str]) -> list[int]:
        action = [0] * len(self.buttons)
        for name in button_names:
            index = self.name_to_index.get(name)
            if index is not None:
                action[index] = 1
        return action

    def _build_macros(self) -> list[tuple[str, list[int], float]]:
        macro_specs = [
            ("NOOP", [], 0.06),
            ("RIGHT", ["RIGHT"], 0.30),
            ("RIGHT+B", ["RIGHT", "B"], 0.24),
            ("RIGHT+A", ["RIGHT", "A"], 0.20),
            ("RIGHT+A+B", ["RIGHT", "A", "B"], 0.07),
            ("A", ["A"], 0.04),
            ("B", ["B"], 0.04),
            ("LEFT", ["LEFT"], 0.03),
            ("LEFT+A", ["LEFT", "A"], 0.02),
        ]

        macros = []
        for label, names, weight in macro_specs:
            if all(name in self.name_to_index for name in names):
                macros.append((label, self._build_action(names), weight))

        if not macros:
            macros.append(("NOOP", [0] * len(self.buttons), 1.0))
        return macros

    def _sample_macro(self) -> tuple[str, list[int]]:
        labels = [macro[0] for macro in self.macros]
        actions = [macro[1] for macro in self.macros]
        weights = [macro[2] for macro in self.macros]
        index = random.choices(range(len(actions)), weights=weights, k=1)[0]
        return labels[index], actions[index]

    def _recovery_action(self) -> tuple[str, list[int]]:
        for names, label in [
            (["RIGHT", "A", "B"], "RECOVERY_RIGHT+A+B"),
            (["RIGHT", "A"], "RECOVERY_RIGHT+A"),
            (["RIGHT", "B"], "RECOVERY_RIGHT+B"),
            (["RIGHT"], "RECOVERY_RIGHT"),
        ]:
            if all(name in self.name_to_index for name in names):
                return label, self._build_action(names)
        return "RECOVERY_NOOP", [0] * len(self.buttons)

    def next_action(self) -> tuple[list[int], str]:
        # If progress appears stalled, inject a short recovery macro.
        if self.stagnant_steps >= self.stuck_threshold:
            self.current_label, self.current_action = self._recovery_action()
            self.remaining_hold = self.recovery_hold
            self.stagnant_steps = 0

        # Only sample a new macro when the previous hold expires.
        if self.remaining_hold <= 0:
            self.current_label, self.current_action = self._sample_macro()
            self.remaining_hold = random.randint(self.hold_min, self.hold_max)

        # Reuse the current macro for this step, then decrement hold counter.
        self.remaining_hold -= 1
        return self.current_action, self.current_label

    def observe_transition(self, obs, next_obs) -> None:
        if frame_signature(obs) == frame_signature(next_obs):
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0


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

    if options.hold_min < 1 or options.hold_max < options.hold_min:
        raise ValueError("Invalid hold range: require 1 <= hold_min <= hold_max")

    out_root = Path(options.out_root)
    frames_root = out_root / "frames"
    metadata_path = out_root / "transitions.jsonl"
    summary_path = out_root / "summary.json"
    out_root.mkdir(parents=True, exist_ok=True)

    games, reason_by_game = get_games(options)
    print(f"Found {len(games)} games to sample")
    if not games:
        print("No matching games available. Import ROMs or loosen filters.")
        return

    if not options.all_games:
        reason_counts = Counter(reason_by_game.values())
        print("Platformer selection reasons:", dict(reason_counts))

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
                sampler = PlatformerMacroSampler(
                    buttons=buttons,
                    hold_min=options.hold_min,
                    hold_max=options.hold_max,
                    stuck_threshold=options.stuck_threshold,
                    recovery_hold=options.recovery_hold,
                )

                for step_index in range(options.steps_per_episode):
                    # Macro sampler decides whether to keep holding the current
                    # action or switch to a new one.
                    action, action_label = sampler.next_action()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    sampler.observe_transition(obs, next_obs)
                    if episode_frames is not None:
                        episode_frames.append(next_obs)

                    pressed_buttons = action_to_pressed_buttons(action, buttons)
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
                        "action_policy": "platformer_macro_v1",
                        "platformer_reason": reason_by_game.get(game, "all_games"),
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
                        sampler = PlatformerMacroSampler(
                            buttons=buttons,
                            hold_min=options.hold_min,
                            hold_max=options.hold_max,
                            stuck_threshold=options.stuck_threshold,
                            recovery_hold=options.recovery_hold,
                        )
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
            print(f"\t\tCollected {game_transition_count} transitions for {game}")

    summary = {
        "out_root": str(out_root.as_posix()),
        "metadata": str(metadata_path.as_posix()),
        "total_games": len(per_game_counts),
        "total_transitions": total_transitions,
        "per_game_transitions": per_game_counts,
        "platformer_mode": options.platformer_mode,
        "all_games": options.all_games,
        "action_policy": "platformer_macro_v1",
        "top_action_labels": action_counter.most_common(30),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done")
    print(f"Transitions: {total_transitions}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()