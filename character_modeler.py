#!/usr/bin/env python3
"""Convert multi-angle sprite sheets into Pix2Vox+ meshes and animation keyframes."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff"}


@dataclass
class SpriteFrame:
    """Represents a single sprite frame."""

    path: Path
    rel_path: Path
    angle: str
    frame_index: int
    animation: str


@dataclass
class SpriteAnimation:
    """Group of frames that belong to the same animation."""

    name: str
    frames_by_angle: Dict[str, List[SpriteFrame]] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return sum(len(frames) for frames in self.frames_by_angle.values())


@dataclass
class SpriteCharacter:
    """A character that contains a collection of animations."""

    name: str
    animations: List[SpriteAnimation] = field(default_factory=list)

    def iter_angles(self) -> Iterable[Tuple[str, SpriteFrame]]:
        for animation in self.animations:
            for angle, frames in animation.frames_by_angle.items():
                if frames:
                    yield angle, frames[0]


class Pix2VoxCommandError(RuntimeError):
    """Raised when Pix2Vox+ inference fails."""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DOOM sprite sheets into Pix2Vox+ meshes while exporting animation keyframes. "
            "This script does not ship Pix2Vox+ itself; pass the command you use to run it via "
            "--pix2vox-command."
        )
    )
    parser.add_argument(
        "--sprite-root",
        type=Path,
        default=Path("assets") / "sprites",
        help="Directory that contains the per-character sprite folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("characters"),
        help="Directory that will receive the generated meshes and keyframe data.",
    )
    parser.add_argument(
        "--pix2vox-command",
        type=str,
        required=True,
        help=(
            "Command template that executes Pix2Vox+. Use placeholders {input}, {output}, {weights}, "
            "{format}, {device}, and {name}. The placeholders will be replaced per character."
        ),
    )
    parser.add_argument(
        "--pix2vox-weights",
        type=Path,
        required=True,
        help="Path to the Pix2Vox+ checkpoint file that your command expects.",
    )
    parser.add_argument(
        "--mesh-format",
        choices=("obj", "ply", "gltf", "glb"),
        default="obj",
        help="Mesh format to request from Pix2Vox+.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device identifier forwarded to the Pix2Vox+ command template.",
    )
    parser.add_argument(
        "--max-angles",
        type=int,
        default=24,
        help=(
            "Maximum amount of distinct angles to forward to Pix2Vox+. Excess angles are ignored in the order "
            "they were discovered."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Frame rate used when generating animation keyframes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect sprites and write metadata without calling Pix2Vox+.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sanitize_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in name.strip())
    return clean or "frame"


def discover_characters(sprite_root: Path) -> List[SpriteCharacter]:
    characters: List[SpriteCharacter] = []
    if not sprite_root.exists():
        logging.warning("Sprite root %s does not exist.", sprite_root)
        return characters

    for char_dir in sorted(p for p in sprite_root.iterdir() if p.is_dir()):
        animations = discover_animations(char_dir, sprite_root)
        if animations:
            characters.append(SpriteCharacter(name=char_dir.name, animations=animations))
    return characters


def discover_animations(character_dir: Path, sprite_root: Path) -> List[SpriteAnimation]:
    animations: List[SpriteAnimation] = []
    for anim_dir in sorted(p for p in character_dir.iterdir() if p.is_dir()):
        frames_by_angle = collect_frames_by_angle(anim_dir, sprite_root)
        if frames_by_angle:
            animations.append(SpriteAnimation(name=anim_dir.name, frames_by_angle=frames_by_angle))
    return animations


def collect_frames_by_angle(animation_dir: Path, sprite_root: Path) -> Dict[str, List[SpriteFrame]]:
    frames_by_angle: Dict[str, List[SpriteFrame]] = {}
    child_dirs = [p for p in animation_dir.iterdir() if p.is_dir()]
    if child_dirs:
        for angle_dir in sorted(child_dirs):
            frames = _collect_frames_in_directory(angle_dir, sprite_root, angle_dir.name)
            if frames:
                frames_by_angle[angle_dir.name] = frames
    else:
        frames = _collect_frames_in_directory(animation_dir, sprite_root, "default")
        if frames:
            frames_by_angle["default"] = frames
    return frames_by_angle


def _collect_frames_in_directory(directory: Path, sprite_root: Path, angle_name: str) -> List[SpriteFrame]:
    frames: List[SpriteFrame] = []
    for index, image_path in enumerate(sorted(directory.iterdir())):
        if not is_image(image_path):
            continue
        try:
            rel_path = image_path.relative_to(sprite_root)
        except ValueError:
            rel_path = image_path
        frames.append(
            SpriteFrame(
                path=image_path,
                rel_path=rel_path,
                angle=angle_name,
                frame_index=index,
                animation=directory.parent.name if directory.parent else "unknown",
            )
        )
    return frames


def prepare_pix2vox_views(character: SpriteCharacter, max_angles: int) -> List[Tuple[str, Path]]:
    picked: List[Tuple[str, Path]] = []
    seen_angles: set[str] = set()
    for angle, frame in character.iter_angles():
        if angle in seen_angles:
            continue
        picked.append((angle, frame.path))
        seen_angles.add(angle)
        if len(picked) >= max_angles:
            break
    return picked


def stage_views_for_pix2vox(views: List[Tuple[str, Path]]) -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory(prefix="pix2vox_views_")
    staging_path = Path(temp_dir.name)
    for index, (angle, source_path) in enumerate(views):
        target_name = f"{index:02d}_{sanitize_name(angle)}{source_path.suffix.lower()}"
        shutil.copy2(source_path, staging_path / target_name)
    return temp_dir


def run_pix2vox_command(
    command_template: str,
    replacements: Mapping[str, str],
    *,
    dry_run: bool,
) -> None:
    formatted = command_template.format(**replacements)
    logging.debug("Pix2Vox+ command: %s", formatted)
    if dry_run:
        logging.info("Dry run: skipping Pix2Vox+ execution for %s", replacements.get("name", "unknown"))
        return
    args = shlex.split(formatted)
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as exc:
        raise Pix2VoxCommandError(f"Pix2Vox+ command failed: {exc}") from exc


def write_keyframe_file(
    character_dir: Path,
    animation: SpriteAnimation,
    *,
    fps: float,
) -> None:
    animation_dir = character_dir / "animations"
    animation_dir.mkdir(parents=True, exist_ok=True)
    keyframe_path = animation_dir / f"{sanitize_name(animation.name)}_keyframes.json"
    frame_duration = 1.0 / fps if fps > 0 else 0.0
    keyframes: List[MutableMapping[str, object]] = []
    for angle, frames in animation.frames_by_angle.items():
        for frame in frames:
            keyframes.append(
                {
                    "animation": animation.name,
                    "angle": angle,
                    "frame_index": frame.frame_index,
                    "timestamp": round(frame.frame_index * frame_duration, 4),
                    "source": str(frame.rel_path).replace(os.sep, "/"),
                }
            )
    payload = {
        "animation": animation.name,
        "fps": fps,
        "keyframes": keyframes,
    }
    with keyframe_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_character_metadata(
    character_dir: Path,
    character: SpriteCharacter,
    views: List[Tuple[str, Path]],
    mesh_path: Path,
    sprite_root: Path,
) -> None:
    metadata_path = character_dir / "character.json"
    view_payload = []
    for angle, source in views:
        try:
            rel_path = source.relative_to(sprite_root)
        except ValueError:
            rel_path = source
        view_payload.append({"angle": angle, "source": str(rel_path).replace(os.sep, "/")})
    payload = {
        "character": character.name,
        "mesh": mesh_path.name,
        "view_images": view_payload,
        "animations": [animation.name for animation in character.animations],
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    ensure_output_directory(args.output_dir)
    characters = discover_characters(args.sprite_root)
    if not characters:
        logging.error("No sprite characters found under %s", args.sprite_root)
        return 1

    exit_code = 0
    for character in characters:
        logging.info("Processing character %s", character.name)
        character_dir = args.output_dir / sanitize_name(character.name)
        ensure_output_directory(character_dir)
        mesh_path = character_dir / f"{sanitize_name(character.name)}.{args.mesh_format}"

        views = prepare_pix2vox_views(character, args.max_angles)
        if not views:
            logging.warning("Character %s does not provide any viewable frames; skipping.", character.name)
            continue

        with stage_views_for_pix2vox(views) as staging:
            replacements = {
                "input": staging,
                "output": str(mesh_path),
                "weights": str(args.pix2vox_weights),
                "format": args.mesh_format,
                "device": args.device,
                "name": character.name,
            }
            try:
                run_pix2vox_command(
                    args.pix2vox_command,
                    replacements,
                    dry_run=args.dry_run,
                )
            except Pix2VoxCommandError as exc:
                logging.error("Pix2Vox+ failed for %s: %s", character.name, exc)
                exit_code = 2
                continue

        for animation in character.animations:
            write_keyframe_file(character_dir, animation, fps=args.fps)

        write_character_metadata(character_dir, character, views, mesh_path, args.sprite_root)
        logging.info("Finished %s", character.name)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
