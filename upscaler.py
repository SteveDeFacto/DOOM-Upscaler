import argparse
import configparser
import ctypes
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Protocol, NamedTuple
from base64 import b64decode
import zipfile
import zlib
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops

try:
    import cv2  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency may be missing at runtime
    cv2 = None  # type: ignore[assignment]

try:
    import qrcode  # type: ignore[import]
except ImportError:
    qrcode = None  # type: ignore[assignment]

try:
    import windnd  # type: ignore[import]
except ImportError:
    windnd = None  # type: ignore[assignment]

try:
    from character_modeler import (
        Pix2VoxCommandError,
        SpriteAnimation as PixSpriteAnimation,
        SpriteCharacter as PixSpriteCharacter,
        SpriteFrame as PixSpriteFrame,
        ensure_output_directory as ensure_pix2vox_output_dir,
        prepare_pix2vox_views,
        run_pix2vox_command,
        sanitize_name as sanitize_pix2vox_name,
        stage_views_for_pix2vox,
        write_character_metadata,
        write_keyframe_file,
    )

    CHARACTER_MODELER_AVAILABLE = True
except ImportError:
    Pix2VoxCommandError = RuntimeError  # type: ignore[assignment]
    PixSpriteAnimation = PixSpriteCharacter = PixSpriteFrame = None  # type: ignore[assignment]
    CHARACTER_MODELER_AVAILABLE = False

    def sanitize_pix2vox_name(value: str) -> str:  # type: ignore[func-returns-value]
        clean = re.sub(r"[^A-Za-z0-9_\-]", "-", value.strip())
        return clean or "character"

    def ensure_pix2vox_output_dir(path: Path) -> None:  # type: ignore[override]
        path.mkdir(parents=True, exist_ok=True)

    def prepare_pix2vox_views(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise RuntimeError("Pix2Vox+ integration is unavailable in this environment.")

    def stage_views_for_pix2vox(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise RuntimeError("Pix2Vox+ integration is unavailable in this environment.")

    def run_pix2vox_command(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[override]
        raise RuntimeError("Pix2Vox+ integration is unavailable in this environment.")

    def write_character_metadata(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[override]
        raise RuntimeError("Pix2Vox+ integration is unavailable in this environment.")

    def write_keyframe_file(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[override]
        raise RuntimeError("Pix2Vox+ integration is unavailable in this environment.")

try:
    import torch  # type: ignore[import]
    try:
        from realesrgan import RealESRGAN  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - optional dependency may be absent
        RealESRGAN = None  # type: ignore[assignment]

    class LibRealESRGANUpscaler:
        """Lightweight wrapper that drives a Real-ESRGAN compatible PyTorch model on CUDA or CPU."""

        def __init__(self, scale: int, model_pth_path: str, half_precision: bool = True) -> None:
            self.requested_scale = int(scale)
            self.model_pth_path = str(model_pth_path)
            self.half_precision = bool(half_precision)

            if not os.path.isfile(self.model_pth_path):
                raise FileNotFoundError(f"RealESRGAN .pth not found: {self.model_pth_path}")

            if torch.cuda.is_available() and self.half_precision:
                self.device = torch.device("cuda")
                try:
                    self.device_index = torch.cuda.current_device()
                    self.device_name = torch.cuda.get_device_name(self.device_index)
                except Exception:  # noqa: BLE001 - best-effort device label
                    self.device_index = 0
                    self.device_name = "cuda"
            else:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    try:
                        self.device_index = torch.cuda.current_device()
                        self.device_name = torch.cuda.get_device_name(self.device_index)
                    except Exception:
                        self.device_index = 0
                        self.device_name = "cuda"
                else:
                    self.device = torch.device("cpu")
                    self.device_index = None
                    self.device_name = "cpu"
                self.half_precision = False

            self._dtype_ctx = nullcontext()
            self._predict: Optional[Callable[[Image.Image], Image.Image]] = None
            self.scale = self.requested_scale
            self._temporary_weights: Optional[str] = None

            # Prefer the simplified API if available; fall back to manual inference via spandrel.
            if RealESRGAN is not None:
                self._init_with_native_api()
            else:
                self._init_with_spandrel()

            if self._predict is None:
                raise RuntimeError("Unable to initialize RealESRGAN model.")

        # ------------------------------------------------------------------ setup helpers
        def _init_with_native_api(self) -> None:
            try:
                model = RealESRGAN(self.device, scale=self.requested_scale)  # type: ignore[operator]
            except Exception:  # noqa: BLE001 - native API unavailable or incompatible
                self._init_with_spandrel()
                return

            load_path = self._prepare_native_weights(self.model_pth_path)
            try:
                model.load_weights(load_path, download=False)
            except Exception:  # noqa: BLE001 - weights format mismatch
                self._init_with_spandrel()
                return

            self.model = model  # native wrapper exposes .model internally
            try:
                self.model.model.to(self.device)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort device transfer
                pass
            if self.device.type == "cpu":
                try:
                    self.model.model.float()  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.scale = self.requested_scale
            if self.half_precision and self.device.type == "cuda":
                try:
                    self.model.model.half()  # type: ignore[attr-defined]
                    self._dtype_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                except Exception:  # noqa: BLE001
                    self._dtype_ctx = nullcontext()
            else:
                self._dtype_ctx = nullcontext()
            self._predict = self._predict_with_native

        def _prepare_native_weights(self, path: str) -> str:
            """Wrap bare state dicts so the native loader can consume them."""
            tmp_path: Optional[str] = None
            try:
                weights = torch.load(path, map_location="cpu")
            except Exception:
                return path
            if isinstance(weights, dict) and ("params" in weights or "params_ema" in weights):
                return path
            if not isinstance(weights, dict):
                return path
            try:
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(suffix=".pth", delete=False) as handle:
                    tmp_path = handle.name
                torch.save({"params": weights}, tmp_path)
            except Exception:
                if tmp_path and os.path.isfile(tmp_path):
                    os.unlink(tmp_path)
                return path
            self._temporary_weights = tmp_path
            return tmp_path

        def _init_with_spandrel(self) -> None:
            try:
                from spandrel import ModelLoader  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Unable to initialize RealESRGAN: python package 'spandrel' is required for this model variant."
                ) from exc

            loader = ModelLoader(device=self.device)
            descriptor = loader.load_from_file(self.model_pth_path)
            self.model = descriptor.model.to(self.device)
            self.model.eval()
            if self.device.type == "cpu":
                try:
                    self.model.float()
                except Exception:
                    pass
            if descriptor.scale is not None:
                self.scale = int(descriptor.scale)

            if self.half_precision and self.device.type == "cuda":
                try:
                    self.model.half()
                    self._dtype_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                except Exception:  # noqa: BLE001
                    self._dtype_ctx = nullcontext()
            else:
                self._dtype_ctx = nullcontext()

            self._predict = self._predict_with_spandrel

        # ------------------------------------------------------------------ prediction impls
        def _predict_with_native(self, image: Image.Image) -> Image.Image:
            with torch.no_grad():
                with self._dtype_ctx:
                    result = self.model.predict(image)  # type: ignore[attr-defined]
            if not isinstance(result, Image.Image):
                raise RuntimeError("Unexpected RealESRGAN native result type.")
            return result.convert("RGB")

        def _predict_with_spandrel(self, image: Image.Image) -> Image.Image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(self.device)
            if self.half_precision and self.device.type == "cuda" and tensor.dtype != torch.float16:
                tensor = tensor.half()
            with torch.no_grad():
                with self._dtype_ctx:
                    output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output_candidates = [item for item in output if hasattr(item, "dtype")]
                if not output_candidates:
                    raise RuntimeError("RealESRGAN model returned non-tensor outputs.")
                output = output_candidates[0]
            if output.dtype != torch.float32:
                output = output.float()
            output = output.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
            output = (output.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            return Image.fromarray(output, mode="RGB")

        # ------------------------------------------------------------------ public API
        def upscale_image(self, in_path: str, out_path: str) -> None:
            src = Path(in_path)
            dest = Path(out_path)
            dest.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(src) as img:
                upscaled = self._predict(img)
            upscaled.save(dest, format="PNG")

        def __del__(self) -> None:
            tmp = getattr(self, "_temporary_weights", None)
            if tmp and os.path.isfile(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    _LIB_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # noqa: BLE001 - optional dependency
    LibRealESRGANUpscaler = None  # type: ignore[assignment]
    RealESRGAN = None  # type: ignore[assignment]
    _LIB_IMPORT_ERROR = exc

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_SOI = b"\xFF\xD8\xFF"
BMP_MAGIC = b"BM"
GIF87A = b"GIF87a"
GIF89A = b"GIF89a"
DDS_MAGIC = b"DDS "
RIFF_MAGIC = b"RIFF"
WEBP_MAGIC = b"WEBP"
TGA_FOOTER = b"TRUEVISION-XFILE"

KNOWN_WEB_IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".dds",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
]

TEXTURE_CATEGORY_CHOICES: Tuple[str, ...] = ("world", "ui", "sprite", "character", "normal", "mask")

DEFAULT_DOOM_PALETTE: List[Tuple[int, int, int]] = [(i, i, i) for i in range(256)]

DEFAULT_PTH_MODEL_NAME = "./models/BSRGAN.pth"
DEFAULT_REAL_ESRGAN_MODEL = "8x_NMKD-YanderePixelart4_150000_G"
DEFAULT_DETAIL_SCALE = 8
MIN_MODEL_INPUT_DIM = 32
MIN_MODEL_INPUT_MULTIPLE = 8

MATERIAL_EXCLUDED_CATEGORIES: set[str] = {"sprite", "character", "ui"}

SPECULAR_LEVEL = 0.3

FONT_NAME = "Consolas"

ARCADE_WIDTH = 960
ARCADE_HEIGHT = 600
ARCADE_FPS = 60
ARCADE_SURVIVAL_MS = 300_000
ARCADE_SPAWN_INTERVAL_START_MS = 900
ARCADE_SPAWN_INTERVAL_END_MS = 220
ARCADE_SHIP_SPEED = 6.0  # pixels per frame in the standalone game
ARCADE_FORWARD_SPEED = 4.0
ARCADE_SHIP_LEFT_BOUND = 40
ARCADE_SHIP_RIGHT_RATIO = 0.35
ARCADE_SHIP_HEIGHT = 60
INITIAL_LIVES = 3
ASTEROID_MIN_SPEED = 4
ASTEROID_MAX_SPEED = 10
ASTEROID_SPEED_BONUS = 8
ASTEROID_MIN_SIZE = 24
ASTEROID_MAX_SIZE = 78
ASTEROID_SIZE_BONUS = 32
GOLD_ASTEROID_CHANCE = 0.18
GOLD_BONUS_EXTRA = 0.12
ARCADE_POINTS_NORMAL = 5
ARCADE_POINTS_GOLD = 100
ARCADE_HIT_PENALTY = 500
ARCADE_MAX_HIGH_SCORES = 10
ARCADE_HIGH_SCORES: List[int] = []
ARCADE_SCORE_FILE = Path(__file__).with_name("arcade_scores.json")


def load_arcade_scores() -> List[int]:
    try:
        raw = ARCADE_SCORE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001 - best-effort persistence
        logging.getLogger(__name__).warning("Unable to read arcade scores: %s", exc)
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.getLogger(__name__).warning("Arcade score file is corrupt: %s", exc)
        return []
    scores: List[int] = []
    for value in payload:
        try:
            scores.append(max(0, int(value)))
        except Exception:  # noqa: BLE001 - skip malformed entries
            continue
    scores.sort(reverse=True)
    return scores[:ARCADE_MAX_HIGH_SCORES]


def save_arcade_scores(scores: Sequence[int]) -> None:
    try:
        data = list(scores[:ARCADE_MAX_HIGH_SCORES])
        ARCADE_SCORE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - persistence failure should not crash game
        logging.getLogger(__name__).warning("Unable to write arcade scores: %s", exc)


def derive_realesrgan_model_name(*candidates: Optional[object]) -> str:
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        try:
            stem = Path(text).stem
        except Exception:
            stem = ""
        if stem:
            return stem
        return text
    return ""


def resolve_realesrgan_pth(args: argparse.Namespace) -> Tuple[Optional[Path], List[str]]:
    """Locate the RealESRGAN .pth file from CLI args, env vars, or bundled defaults."""

    def _unique_roots() -> List[Path]:
        exec_dir, resource_dir = _runtime_directories()
        roots: List[Path] = []
        for root in (Path.cwd(), resource_dir, exec_dir):
            try:
                resolved = root.resolve()
            except Exception:
                continue
            if resolved not in roots:
                roots.append(resolved)
        if not roots:
            roots.append(Path.cwd().resolve())
        return roots

    search_roots = _unique_roots()
    notes: List[str] = []
    seen_paths: Set[str] = set()

    def _expand(raw_value: object) -> List[Path]:
        try:
            candidate = Path(raw_value).expanduser()
        except (TypeError, ValueError):
            return []
        if candidate.is_absolute():
            return [candidate]
        return [(root / candidate).resolve() for root in search_roots]

    def _select(label: str, candidates: Sequence[Path]) -> Optional[Path]:
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in seen_paths:
                continue
            seen_paths.add(candidate_str)
            if candidate.is_file():
                notes.append(f"{label}: using RealESRGAN model at {candidate}")
                return candidate
            notes.append(f"{label}: RealESRGAN .pth not found at {candidate}")
        return None

    def _finalize(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        args.realesrgan_pth = path
        model_name = derive_realesrgan_model_name(path)
        if model_name:
            setattr(args, "realesrgan_model_world", model_name)
            setattr(args, "realesrgan_model_ui", model_name)
        return path

    cli_candidate = getattr(args, "realesrgan_pth", None)
    if cli_candidate:
        return _finalize(_select("CLI", _expand(cli_candidate))), notes

    env_value = os.environ.get("REAL_ESRGAN_PTH", "").strip()
    if env_value:
        return _finalize(_select("REAL_ESRGAN_PTH", _expand(env_value))), notes

    default_candidates: List[Path] = []
    default_candidates.extend(_expand(DEFAULT_PTH_MODEL_NAME))
    default_model_file = f"{DEFAULT_REAL_ESRGAN_MODEL}.pth"
    for root in search_roots:
        default_candidates.append((root / "models" / default_model_file).resolve())

    resolved_default = _finalize(_select("DEFAULT", default_candidates))
    if resolved_default:
        return resolved_default, notes

    auto_candidates: List[Path] = []
    for root in search_roots:
        models_dir = (root / "models").resolve()
        if not models_dir.is_dir():
            continue
        try:
            for entry in sorted(models_dir.glob("*.pth")):
                auto_candidates.append(entry.resolve())
        except Exception as exc:
            notes.append(f"AUTO: unable to scan {models_dir} for models: {exc}")

    resolved_auto = _finalize(_select("AUTO", auto_candidates))
    if resolved_auto:
        return resolved_auto, notes

    search_labels = ", ".join(str(root) for root in search_roots)
    notes.append(f"AUTO: no RealESRGAN .pth models found in search paths: {search_labels}")
    return None, notes


def lerp(a: float, b: float, t: float) -> float:
    clamped = max(0.0, min(1.0, float(t)))
    return a + (b - a) * clamped


def _runtime_directories() -> Tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        exec_dir = Path(sys.executable).resolve().parent
        resource_dir = Path(getattr(sys, "_MEIPASS", exec_dir))
    else:
        exec_dir = Path(__file__).resolve().parent
        resource_dir = exec_dir
    return exec_dir, resource_dir


def _init_arcade_scores() -> None:
    loaded = load_arcade_scores()
    if loaded:
        ARCADE_HIGH_SCORES.extend(loaded)


_init_arcade_scores()


_QrCodeGenClass: Optional[type]
_QrSegmentGenClass: Optional[type]
try:
    from qrcodegen import QrCode as _QrCodeGenClass, QrSegment as _QrSegmentGenClass  # type: ignore[import]
except Exception:
    _QrCodeGenClass = None
    _QrSegmentGenClass = None

# Embedded copy of Project Nayuki's QR-Code generator (MIT License), compressed via zlib/base64 for distribution.
_QR_CODEGEN_B64 = "eNrNfWt320aS9mfpV/TKmw3oUDRJURd7Rz6vIiuJdnyL7FxmtLIOSIISIhLgAKAlenb++1tPVd9wISXbOWc3x4pEsNFdXV33rq5+pDYfqZ/P1HE6jtRVlERZWKSZmsbDLMyWKni7LK7TpEWN6N9xOl9m8dV1oYJRS73N0j+iUaFeh8vFTdxRwavT9+plPIqSPMIL10Uxz589eXJ7e9tJpE2cPpmHV9GTf2TbIxpw2w64rQeUcd5G2SzO8zhNVJyr6yiLhkt1lYVJEY3bapJFkUonanQdZldRWxWpCpOlmkdZTi+kwyKMkzi5UqEaEbzUknosrqmjPJ0Ut2EWUfOxCvM8HcUh9ajG6Wgxi5IiLDDiJJ5GuQqK60htvdNvbLV4mHEUTlWccH+RMl+q25hwtChUFuVFFo/QS5uajaaLMeAwX0/jWazHwOuMyJy6pe4WOc0D0LbVLB3HE/yOeHLzxXAa59dtNY7R+XBR0MMcDxnPbczlCS1YHk2n6CEm2PWMHYTcChOYA7GFRhXGVrfX6aw8G0LUZJElNGjE74xTQlybeqRReb3pGV6YpNNpeosJjtJkHGNe+TNqtq3e07fhMP0Y8ZSEYJK0IIgFDqzF3C2x/iq/DmkKw0hjjgZnTCsVejPLAEVeECXEtBTzNONhQQ3+HDoWDH9W8yz9GKPfrTCnz1ttuzLUAsQFWmFSuokTorPobk4rikEZjHg2n8agP7eytBqAXlZWsAUwdHeyFFhIolSirmE8jYtlmzubxEWCvic0o1DNQ5rHaDENM1rvbJ7mgqgkJTqeZDRSBPLsqFMgS0Uf6YNGFw0nOFrQTDJGkMP5dTod00oDp9M4HE4jGY4mOJqG8YxoKpwRN/JbKfUk80RTgZSII8JjmjG9pUKmbOa8NCky+gimyAr7+m3MBJnFOZAzydKZTBYoprdS7ojeTSLpCegv0x41wWdiB9spMx31l+NlM11/oTc3MZK6vJwsikUWXV5ipQBWmNDaML/lm/rRiGhWBs9pHYsoK9J0Sn9mkXTifd8JhyPT07voH4soGelGxXKO+env3szRPJy21S8J/bG5Cdqj/6xQzZezYToFymm98c3mpvz9c4bvn21ubG1tHVWaA/fx6BpkG2JAFnjFbbo9jokUmG2GYQYZ2tncOE1AEUR/JCVf0Lep+i38KCQ0jvIRSQ1mJkbt6bs3T05PjlXvoNsdKHDSOMzG3Ak+jCLNTDSwQJlF4AJQHJFAPJstCqak/B8LLNhVFo/xAnVywwNOmfJGJI1y6hQsKN1o5ssxJNG6mhD5pKRgJotE8A3mGWURCWSasUGG4Du6YxobxglU0jgswlLXI5I0ROWYnXnvFf1vqvoqn0ejeBKPQpHJ+WLOIgPKgZgHr/HQQR5/ivLW5gaP1wMog26b2wxUlGXMVVmmyXZKDAiiwXQHrIVoLkSoRCEpiwUS4RFmv7nxW7hsnljKopTWflv9RBiTPp+p9+FNxBOZh8tpGo55sjzQCMAIxXR4oOgSeAlawEz5ueApaHXQ+ytaH9358SIv0tn2zIxBeoX5Mo+uIF7yVeOY73WPL9PbVT1C7i2FHApo33F0m2YgywIa1crNzQ1lBiXlHrKMwtgTApyEejhGI8hfI2drS2C6zltt7ozWdboUEOZEafMMit2sr0oWs2GUtd38mFR4joQ/EkikWRcgR5pf8DotMJmw4OW/xQJmxP4xEbsMA4DQAa0xPRyvoI9Oi7gaJKClwbtVZB9c2/VviXigl/6fcMmMxG863twYRxPlLzr+94w4ieYUjWgd9HqdjMgq237uBAsky1lEUjHJPdqzHG1mchWT/ID4wgjCbrA1wCWF12DFRGmYo5yNLbJHso8EN8kewhRxxDBdQJGyNDEzhki7ItkBY05UZr4YjfAnlJOMm8sCXEOK7e8cgMQn0S3pp40NAyX/b57GoNsAOviX9z9s9/bk+SKJi7ylYrEKyEpZATtgIV0z7rCtkM9oxcmEo27JMCEhR8MZnBlSgjxeEM1jdag1adJr0tYJw46xSNXNF4X0d3J8rIfR5glhfTEt1Ix4hFQy1j3CW6HotY0NWktioSu2RQE8WWoj+pLajtPEmZnERyROcrN6GrQO09sG8VX+jFn7/OfsnTDZhTpU9kMH3Op4GqtNgm8jYzJZxfnolWmtBepcR59a+EAAPBONeM7s3zY69JyW7OLizyRcTy88lGDflygynDKbywxAEaZ/3bORVZDrbTb4YEMxn+CRrPYsvItni5kWNrDBNzZE8IWwk6GBc9V/uruzgtjUF5Iaz+ZziU39GaS2nmTOKxTHuGC6aF34pHSvfJwZDfYQ8Vii2GfWcPN4oUZ55GrFiZ7ZM5pxQdzSa2M9Kw9hEszC/MZ83qZWwzTNC+6Q/prSw/fZIvoKWra62NrGmpSNbUGuQkjzJsjMyt9PSehLm3/SGwnfq6hGYNTfCml2OpnYmeI1Upfwu68j6XQF7VF/fyb1UXeG/jTD5TeG2wimKGZvYRgVtxHB1YVm2ddGR06zbuEBTWwk2p2By2/a0C3bbPHVmE3csJJFwWMGYprrueXE3K0mmULPc+PNZCUjcMRmE621EGfJBmPFlxMGRtdRDhGi58M2JRmqCxo5hEyZEzIJgRkxDewasBdPURhBpFMMkUcWcj4PR5GFkX0K4qptWbOjt6f/yXBildwzxmnJ0MQwVRNTJAH9o1WEYAw0X706fX3568nZu9M3r9VfDj0O40+WtfDJvHH0u3mDbVrujVaG2xPW6fd+C+y0kYXkZ6pfw+kiOoGED7bIASIfcaw+4tlWSyB6pH6IOdoQYfiYuKRiEAJbtDbUFPRuxa5mj8DB7AsD9Z3qCRgQZaOQMEuu8pDsDSMW9Hwur6Likga6RLtLa7AGtk+IQPVYHSj1SL02SkO0GLpT4ccwnoZihvBgsFRkIONzsi4tKXcMWpDXO71ES6239ZDAC9Yp8DvDMgPVr4nxZInLXxPaqxOV6W8MiZxvADwTVQW1YAUYf8AxmGQRF2YqBIFp/PywJGapryPfLdMyS6TVKF1MxwzqJPaVPKBjeGb5FdvEhI8tjQ4aPSXzL7na4haxoLdp6jIl9EGvT7ZeYBWmUXJFQvhQ/dN/61+8Om31KrxTBiumjY8laScDC8Wi0/dp+pLgEbql0bAm5MNGWbESNEPNpyINxdNaYc+SZJpKA6ajvIgJnxPuUHA5WtAbhJjycmkWSMjGhoBPLBuTeuy8Onlx+surtqcxOz//cnT2/vTlSenhT6c//tTiRfwBXjSsblp8CBW96laDNFLZQ9lGgGTO0YSILg/1c4Os4zQhpz9KWOASDpx8tXLYYokg0H4H5O2Q+rr8Pi6+X0wmURa0NG6oAyCGzQqMOhx2SDFEydgyGvPejH1K+6d82WqrQWvNS5guwgjcstKT+Y5QsUgKec/ws+mTxDP1yd2hAdtYjq6IjIPhsKUOD0skZjB1NCYmRTA24cA/1oY8cHLgWCeyXoGKJrinpBqFh6u0/gVSrw5fg6TZrKGsy9ZaMGjX2qpt0xPjpf7ethnoG6IcFrrk6T9TssHxbQ4Nu5imCkwRXYkXK9yBACpcUfwuCmPHHJOtOovJUpiSeFogjFqfEo0EtHcNrt8SYtmwC6fUDyGcg8jsJNDixlMdNzFSJUboISRDYKyJkBZG1iNxwcvOaDmaRkHQvTs5bqvuXa/XEr6gRTNwPK+j1pPhDZSpB2oTphzsoxtRTKR0Ug11DO/riqyCcRwmhjDMohNVoBUHhYLz7gUte2AgevKEuja8FcRt9NxCdxEbNcSfaOb0rOnzPFbPn6udC/U/h8y3f/mLCvZp6YNY/QfZBxbaY8fj5ZBb1W8pUaamKjOYmPtlP+VtFn9Ez5M4mo5z65E8YpO0ogRNIFVD4MV0jVXXkyBi17dTO9IbtRpHwpiR2JEIUto+XfT30ndTHCy38RiERt1fRxyTrUEzI22yyCQ0zMRPXrqBDL30BbrefsmM7ihjRJL5SjYV0YKZ9mM1gHm0D6AAbAWiFToLAknUkwMOPVBzajv9WHLWXG9k20V3xuFgC3EegjuTxg4bcN/l2e3XUH8CqwISz1IOYuFMT4Z7jafA44KHESEkF4waBAwKnEOEJh957hDaCQ1qvXwdsiGO9my4ViEDEqy/6Saul6q6nir4IZySdXAoIfg2e6LqkCPzMrFTG7wPJxBjXuAToVds+OUddTQakcuAyNx1li6urpVWRjQmB4D13ybIxP+D73txISCekiyA6vV8IQOxRFWziE0bvZsoIUCNx456EeejMMMm3S28y0YQCYY4N503g2HZ9djrIJiayLWLJCB0cHlJPReXl6RApxNrLTPW26qRCityYn2ga6ZXkKMCxtYkv0lEFEiATBe7ihW/vxLCRpCwkYvaTSF3iXh7fnLZBSR0eO4eL86MLDT2V8lnvLYGN7bjxjHGmy453Fv2HjkiAP6ubhdYCnBuIknm64gUCa0ydjtNEEDi/3lUaLl6r0O53n9c4Sb+ql8SbtO7kuxdsM9oRtRO5z0+5yvHt7WeOA47nXSMaFYWYPsNC/PDmui031u6o0b2b+cKxNj/RhfDlOgFu3C5cbXo6WPunRUvy4nviTEislkmkBB2CMOW9N+hOj9n8QEl7QEI7XzpXGL3TesCFpSGg4wg2Ngsd2zvjkW/pHdnx8/mi4JDTETiWXhrhIkdBw8vzVCXWgfkbLcTUB6PisVyqAEIx+PLaDS6JLoj9qd3CEEfyeiwBkuJw9mw8MZzhq0/hrU+XliJJmQFWjqEThAftynQNEQEDy+B2kjnky1GRuLSBiRh6ez08SWwFjusGS9ITws7VKw1grjlPRcspRmNKxae/lYPY9ECYa+fXeZEdFHQMq6zafoX5YFn3GaovFh/sN/SM/3XKgiBjl+ScaotHBZVY+IoIuXf35w5g7rrc6QjYFG1M0ZarXd6zP0f2b1B2V8khMcSdNPoXoEh8/4b4tHbjOxtlU55k4paKOdBYaO5SvElFSQKlRZNwtV5SfcA31oGMP2zkqAlr8SNfUX/bV7b2CwFSs575G52L8pB+pI48gcHu33GyGheHa9PA5KR2Dgi2vvDsfa6dNrrkrWIA8Dp2PVwrNjaaYDAE54ODCaQh89aIsXlWZNH2TxloSpvLLGdxLi402bF0pkEsFpqwyPeNU0za99yHyqYx3ewXkq7XKOUZA+8dyh8a+X+oEX9hJP52P7PxCTEExiFEkUv0jnhboLUmCwhoxBGqQxve1XB3SHNdnnY5Tj36aR5cLbstCrkjd9cbxGIXRowFC1xajHNaFzGXsBMfkfyxVMEEivib5YN35QU2fny4vzuotFbu46m2I82HDhJS9bvM/WCmJ9zlqoma8lOXKFqLB15xt1ZFI41GYnN73Etmzhim6HHXFtp2Q1vFdZg0Fh6xECq6zSLP6VJEUoYjfpkRaKKeCbbQwLUZk1PeKhjqa0/E4HaGWlK3SNSp3/qG9Xn6EVrfXNqu1dt7uDdgdQdR5mFTAWSzwdqy5HjEc0LWDHYEeG0tf9UqZG4SNmcRWZuGh8VVSzdm9UIdtpqZ30LzwDZVoN7m++0VfmN8vw4XJEusHTxVcJBZ28N+CF9nqfGX9IRe6dt7XtmxEtqLNmUrHtpAG5iLAFEULxuGfibeO5vdhaL+ZTdjza7IPRSgABYt9VW+G16pMn08Mj/jFatOvGYJkI6+PKPVV/yDgPRxB8ttqYRMxXwFOyi5NtCzDidBEtOZhQZEtFEIWaEtyI1FPkIOI8v2sr//MdFeYWI0Sfx1SKTzFu9XbBC53dZ479YzGZLzzF3FFmQOJuGhaREipi0UqTcq1HqOnrkyQ9vONEJ1sevCpEXLB2K29Ql9urMW2uDqIAdRvyV3ibNiVIcDg3h0qf+JjDPz+bjWim1IkrDQsuKouNwinTVImoeUUeRdcBQxwUNAVfUckdmI9sAZOOq/xHIaB2ozTF1/PKjJOrQ633Blvm4w9pjZnrWa1txK3pd7cJFM3ACftE4vZb6QGyBT8+fq6eISXfvdnf2GVnebh77AfxClyCj5niPWg56fYAIKHq7zlDlaVCHvd1S4JcJcRJnecHpuXUGI7bcu08uH7BcZqFBwwSyDRV7rsmKl/ZrL+3d/9JB7aX9e17ab3rpoEmaPCV7cfe+2fYGJI/ipj7jVpnB8wgJ6KsRe3DfUCX53ls/bNX7eshUDioqhIjjuy9bylI3B22dbcL7ppzCBLuuKnBq7sUDJIwxVR4sYtplGbPG7IknE7VfCeEMulqyxJOyp0Im375mXtiINirwmeLHkxHl7glzXroVmHmvaufvszfVIFb694uVXk/kSu+H/m5VsFTmSa/2RcBYsXLQIFYO6mLFrV6dPHua9ul9m6FUozo0qEhnQ6g9ECpZdjvciWkTY/tmZz3Nh201lF2d9e2oUWjaVRRl3Wxr9J9q5Byqp4+fVsxO79AGJ/aR1xJhUxfpVKS7Qb428jpCMn1mfC5StMEdDdjqIKucnQKdq1TydTQBs3O1dAuwTQbmrrOaxneNX23cYQTES+4I4WN8wO+ltafENVrjG1WcIx2X4WjrcJlfRx+fxAkC6MWSrLJs5ptYTasi8HDeSxAO82B8R4Yi/7FsWbsu6GNfu7ZudWvtc5Zu9/Fu3ZRuq+bFAYO45UHuiPHbZgtStXyW6J716cMBaF4f99U6TJUWbBXG/u2QJIOHqFVaqIolYs8cMl2Yt4qzd1FRiRWElmqtSxkXnC9WdSvhyb9JpkvZLRsuq9ZsRx17rv0qdGrxhAMqgUDKDj6ArYaZxTuH9MhFS9XixLYFtNoXufHHduvZbHyA4Us+/Iq4ryBfzFQbA5bwDMLGtTRObNnwVrxOlF+RzsnGo6VdP5OwqriQfOmg56142RqLiQju0aycmLcuIFZJZK0EA/2sBc4dgY55aD5HxZpv1VW0zVg16RTi1A6n6eimljT3+pdXlydnZ2/OLo/fnJ2dHL8/ffP68vuXb47/+u686jgIeU4vzjUwUNLcK60vzaXa9cnxMXX64uS3N2cv3l2+PTmTfpV6UMcknry9hOZMP0g/xpMmeZuow+kOMu38Os2K8twtMkjj+sOQ3rVfgV3sm97kSu1pFP8FXod382ksuWU6b0NG4hOwTGWcvYu80XB0Ld8aNJqQBXMA+PKcEZGP449uN8XggNz48WVOcmhGEm0kuzaX1DLO0yzwVgUEYjOou42RBhncJn94DKl9vPMb9UzdkMAtoYSw541D35JqRG4xacYy3lWEeGSPN5k2qJtDQ/lsqtDrD5wd2WohGxl4t60YMSbHsj6qKBJqqRNtAomryZfmGX1NgBMIXv4Wbx1Z3nRbgEZ2yfGXkUt4a4mJw1PgA20kCbKlIEeIIFRIqsYpPnOwEXYsnxfwc3Ya/B3O32GQz7sXLac4gz9ISk1vqik83kqCFm/iuT7dNh6b1CeOEAFNyhKujiFBa65ZYc4SDv5AclMZ08b6kRkZ1BJ05/FFJS9OmrDA8znJBaVNJnvZxHFysKo38hUelpfg75K+D7aRwuT4N7Cn/arSnOaa6MPF0DGZScJDzD2sZoJ01A/VMHaij1oNI7YKoEyQ0h/R0kXyLnJccI5gXNbsa3VCTd6VFIuVevDrLMfDu/k+LnT6TmxmpUM3vHXKIa5FkizVp/jqU3iFZIFEk6IcbG4IauvgZVtt9yUJ9dTkB8krZCItZpzdzYJOf5yHcaY5VpqRLb0n5COft8lwMzSOWHtzPB3j/Woi8ZyoyTm1tUhp35DmnZXfMgo1ghM/KpBRdeftqkjzxfw2zMbWgwvkLWSiq//QUXdpaPeL6zENQM9H4qQzEYI8JTfysjYyGJGzdRptNe2CQNg5QnGJuassQOuBsjDXGX1tVcnj4+EhnXvGmTmtZDxxAhWRiRXEOl4hZz9avP20VLcRKj7kOfkUEco+mN66Tzgd4onsjtUtYNGR9C5vj9G8r7GG2lYmKMzpI8MqcUlOS46rMzvdvvQ9Qd/f35wZo17n8hgmrqS0VVOF/Bw4czaotp9lDHktBUJ9QMLlDQGBphFkXcJnUFhOuCQt2aGXE78EBGEiHoHVCPY2ixEId3/OLQaWuuI8RjIH/bS30ETwiniGWZEbRy5GajMCyLPTW/e30XS6jXgxZ8QaPEC4YScpRH6SwgEiJtlPUZa2ER1pq6gYdVo64D3nKg4u3sQpP90HHTNZl/KDlyNyQSoVBDpkB/MhjnO7H9O2hzZEmr46evfXy7dH79+fnL1+d45+TLRpuXr3Dl/frf66kfE+HEp2Ihks7DHLbp2w8Boe96i4nh3SsIduLX+xMjNvT9vkhfC7zqvBab6oqsXIvzGHFcxGqJe95nut5XRMk98RTq9SJk4+92VPAvmJopzytpREXn1QGFk4DJ3dm4b+9+1Vm9haCpaBAlblRdqGLnVJZ97/EY68+fFyprc49gz+YTZhz75tTseTiNuexjdRdZvXJxRLAxktI962asOkBOCbO39KeEB45YOXhyUKHsNa0bnb+9gCaKI9R3VIdSrLetJwBgxtlNHYnlCH4sUTarhrlIa2RL9zLPL25PXRy/d/u3yt34qm5r3nTa9JKyi4Z36gywQUNQEjEqBnHaCvtnJoaFnowBiVKXgjNXYsJzVseoDXK2GxNqUdixeJ4hyqSjqDxZqe2D2jm7McEYc4GJbAdM5TLE90FUjN5KnNps+n0Lt1FFqhzeVX0ubyK2hzWaPN5RfS5vJPpM3l/3HaXP5ZtLl8EG0yefYf900wI7VLW6fMZgnJqQ/PVggz92Uj0fgPYIOXH/KTakv9kH+VnNNGQuqbKX4fTkPtKlbq6zhNItFhrY4WsyDowcZH/R2x8EnBs99EesUd85DIKdqIygFJ8alR2xOSiB/3zSkpfSoZuEH8cTxuqzxlmJ7waxzfftIvZewW/tF0RFqvyD6/gbfeVXyKmfsMBrvbuzetb+T0l+2PPgW7u9/hGz9eJJF1oOKx6ndpnaT1Y9WjaX6nP2H14HraT5U0Urbxnm76S3DTRGmDynu6Of3V3907oP8EO8l2wUviVTxxlg1ZTSFeYqORCdWJi7bqdDptxeM0RBuaQ9+uLkIpqn1fIpE102wOUjWinZBnNMI5LuIeUxjIvs9bC7X8Jl0moHb4F3baCZxs8z6Ippo82duHg2ZcrIXGF2ezo90df7VU4Z1n98kRYGxDzOZTLonGDh3ZbekNzivy2RYCddAlhGdc8WxbH+LFlHgamhLzVRvOxLY9b8dZYp5WXFeTscrvEtHtExlylnZeRPOKM+5y/Q+olc26eqx26OMu02zgPR1wshko3UnWchLZednLh/8cI8Oehhaub8jgYva4oAHP9y68aaLbIEOgkFbChMVWF3tZG/J2bm1TYm3knY5zh91ZGOh9VQh/OcHl1cuarD4UI8eZJN+/7vGCvKSSlNT5YVrkj3klcMBCLSbhagoWApiQvOFpEc+Fsg6MY62FQY2w+92Dtuo/3RscXOjBVhFtSGS6TZ/IrvHpt+KYrjz98mUnXwzqq95r2c0JentERhiEFFb/oGU/7A02zfl9CPJ+I0/gS58TNKq2qd/+LnXlU2K3VX6wu+sqBGCEfR2Ls13s7DmpTLiuyGXgvVmarqPh+taWOfvVUKLoPpKWmC4TdhB3og4vYz2c64K+qKVIRKgJPll6gY01RL9mz09vlleCYmNzsM5KU/a6uTzIfJF5oaLVBBsMuo8HrW02HBpotlwPaK2IkNiw+m+s7vb9O3RKndNalPblLuTlxw/aOay9vJoo1u5jjaMr+t5L3W/cGz6jLrbfSRe8u+bK6s7T6TJJZ6hdamrs6I1i7rkiMsBs1TUo6zqwCaSeLf3jNlvF5EA2E+mDkIs0hqLnTWCkImr41JvAIUbO7ipp8kIa1aTII/XWzW+URpNJPIrldF9mxTrvSHFBoJxLyeqgy5x+ZX7NPw7IEOD8N/kNXkFMyXXrdaTIC+ExuguBJ4kxORDuPkCz0kTuPvTpjwMY4E93FNM+w6LPOiDb6kAXLzyn5pz993TnotO8LaaPsmtUsblJ6rR3wYl474oQFVEnExeeJTIw8HSr59oY4owYY1T4kAd32Av+wOJR/93z/u7jb7Ie3ZN/CjTbvX9xKt4jfbogndvKPny+zEDSjNC7D5oMuQdn+MPx6951+6WipI6oI6FQG8DjRD/14w9B/8PBExQkeMHnRrI0tRqmV8+nk6FlL+WXhA1CY8TJybJH6pVo4mWpkonB3nBpcBG3GpLUTfeePjn/42LVFq/W+cvAtiTHlOC3h+H+gGpUf9H8UvLp0O0Ho5bOuaFYWvT+/eNRozYju/VwRXb/7rTZnUQZaRZl/m5lkxQDhktybGUJzqogM1uYeiCPqvNSTLWBpXjzRF6zubZDLKHMQJXkCzeU/JUNqRpniGuoPmiUdebpXO+zlzeDu5ZEcFQBoqqyc23AKNFLzAu7fgHRV1uXsfvSJbSdNaTMNdsfhgm0xYCUUFkO4UXNobbGSZk3pV536SS2LeDGzlmi987EtrEe1H32bdUtI7f5Mf1gBzvTucYseZ0eIgZ+/pzsgn/jjQmsztJ7sEIbfY8kggZddLYgnUj6bh6FeYgwpmBVFwOmJp+a01CsE6SP18q4n2AXf/JSfD8Bsn2d30tolEbYYxGgY+zNstC+cybrJ5mO3rDVtPHJ22FZH66TrUMXIXtWD4iyb1jbkUnIuZ5yuTrZ5Ech52iMRDSc3eZqEKEOLVHvLODH40gOqVlAbY29bht77YSwPrbmKpEJOdXdcI5YL7NLXbLTOGcBqVHECeGeX/uYM43Rhd3ISNRzXSPDC2Ce901I1zwYVB/s8oNEtry85zvyXA9ljiNyIC3gnapKexJWz6X9oPrVnnzVssE3WKkwYJGPtLrHvdU9dis9riaWpuiqUIzWleUdIf+xREm0tPls+nql94s1benTqcgt45TUaYwtWUQgWYNgB6HlxWy/gIYQ4axOCYXoDAIkXErfbT4kwF5FQy3YXm2gRUbt8XclwlVSW0unF0jCuWxFkvYS2L4atNLB4weH+leTkD92lXS+hEZMXoNsiXgkDQn4rBG3D0JiLIUnNBpd11q/I2OjhrtWvTBMaPQd6ygXU/XCLUZakbWndP0dFJNczKoOua4t7yrt12rR61r71L0L2LjasrZ7XUf4y7vf1F4RyrZQ4yY+akt5nQhOolQhMzVN5XAdqlGgzo6LVls87PiP+82Pd9y0/McDa/53AeUqf987yVqupQMVzKr9V5PPjLwF7aRIQllXKk9mroo8lhceX1TIRTEqJmq9QjFQ2BBiKpBCg0ZT5OGoHfoZ0A95gTjXjK1FuIMKh9ag+FDsgNr1qF1vgNNf9LOHAgj0g9Ng1K5P7VAUoU/t+tSuT+361K5P7frUrs8xwrbaoXY71G4HJw1wcBmnkandDrXboXY71G7naZuJQ6mTxmgPwR+gEDHgZPh29fh7Ag//PZCx/M/4Hs8YxoHAxJ8Pyj8Mo+7P//x5P0RwRN4v01sDLoO654E50Gg80CD1KyAaMAceeHuVn4M/60fAfUUW0mJmId4RKHoHFagNxF0Puj39t0FeBWqD6CqivhbBPy9wkQyfixGYDbH1Nbb7FcwZuAauncFyiRi8xTffff1PiwMmP3GBUJIGawJ4/ysSAQzchjjAOS2ucSAnXJ+2/y/Lgcq/vvdvoCWb/7NX+THSTks8K/X63s+aGddmzTOuc38ZvIH5t8v/+KcKgEG5GdwMvO+JtZ5G864nD55q1O5olGp0opb7AGtLzwY44vm0kedrQO7xPzkfjR+LnD39e9+TYz0N144H176GaeAtcxUe+rxL7+7S513qa5fa71Hfe6DCgxWcbiD1cWkhZcz1NGQaOv79VEO56352DH/3HcYYun4FOoxB3+3h6DpBugcGQVEcer4PTulV+LuU5+hx9AMzJg2vB9NwNhyH6o7DIooP4C1bXH2k9h/HQisvLJubrn7hDudfP+cFB9LOg1/AgVq8hT/6pdk0g/RYz+M7+/fOPSM0vKHHaXzBx2v1FbxQqYrKF9IZN86E0SCcjEktVzXpuk56L0bMQaKRQN5v4YIzt5nKadw7KrDxp74ff9IBQl06wusqHN2EV9H2XBInPrtPcWnX1Cst7UvLbV0dV1vTlsgwiRvlCiEN9SZj7VJNhvV8cPEQNbpMiTV+5lXNOKRXbZXeaTiK9L1v9ODlm9+YJsolVDekpHftsSnrXXmMwt71TrD2etlHxpfqeHV3Z4jJ5fFY9jh4+cmjQKFwAxQBTh9RoaHXMt6PwStSAYp0ysFY3CW4oMXd/4aXJIlQbsc7q+OKlNs+uY7NA/rs7a7p06BD98nHgR/QZ39dnwaXuk9ohYf0udNt7tPdNscV681FNU13zemS+Oa6OdNU36PJN5g9kWsdnvD1funUnBqsk/vq2+L4qkXLC3JLmysWehsuSxdhGCBiLl5ZNF1+trlhb+0KV90dp8ytGJUrZ/T9GFy29r3kdz8IkFHlWjNIiCGXg/egkTvE9Hjla8R0lWD/JjLejGUw2PL0lwn3COY4o5WaGwe8e0RRkZnWX+M/lyux/O/BU1ywONF7jXlB+PmYZhx1dzdztr1VHJl4MAQFKe2Dp44GcpOXQx0fJe7uI9yi4K5wkST8WRQmfI4wz+0ukLm/Mp2YHUJEFwz8nYdeivawS38qFwutq8Orqws6Lijv2ZtpPuCKKSlzy3EYPr84k3rVG0ecTjlfFPJYMs74ttkRapwxQ6gjc3GUPjeuU55onXAUiw/p4ZKz9z9sH5hr83J9tVKwxY+3Wv5NLHqrxQJiJqLV0KrbDNyuWkPx96Eu+27zOQyRO/ZC0Knz/d/en7TdiSei1OHqjLESQ47jK9bZhISvWxeNRZBsNOIrXqRrf5HMLTm8TqWUBzefOK/Atmqz6Z1eNknaAdMm22YAx0NbrRW4j/0tJ7mswx6e0+NCGyA4SZ3qSxh2zKRgW6Huv9nAcnspuBPB60RKGO003TlB7XWr85g0eoxMxAtaO5OH2JMTzAjDJg+gAXLaT85Oj9v+FB5ACP71Re42w6+jBZ+tvOUv3ZRkeJUvD3Uiz9zQFmKTiU1EcoyO8PvvZCoisXeEi1cgM3FDB25Tgs08Js8lzNpYFxyUIBGbU3cxaoXOpwsSudfL+XWUcIM4RdI0CfzrNiemJ2tIsYadh1LjIuGJi+h302vCwkoSrdfPwtIyFHLGVp+vfZulqGurrrJ0MWetwdl+RTSbu/oIZlKXRy/f/nSkieXy5PXxmxenr3+8fH/0/cuTc/R9Hl8gA2Cwa/qQzPjPeJ/zLC4aKB69wf3VJb7dbODaPFddnk3Pu8dVMiekeHO1s88BabuH8u97DxGlfl8ewh/ASaWLHB0XVS6AbKxfYm4DMOneODCJLboZNij9C3PcFcBrbwut5MbifjRsQSCLB1arf4uirvfO15zpBzROOkelz0/6dlqxhXRGg76jJwpnfi35d9GULyQ3ho/NOLOD2Xvzqjfd4W5QgH54qLa2GhK+6ftG9eCxo26/wujkhs19reLv5g7rrUsp6SveEqsIzWvmw8V6iiJFGshx6Y/aRf8SuUx2zQlfS0Ri9ZjgT8jq5poV1KYwCiw4OT5t8UW6V3L7UOVIs4DBnbMFvcaqwYUdBmjSqN0VEpMGrPVaTx7hJfO7C7j+uj4YXBEJtp02m1a83Bs0vt0dIoTYb63vuDdo7LnX5f9W9IuOd+7puN9rlYjpCxF2n3yjXrg4gxVn2KyHfWxumK+ZdNowLlm51kurGHXo7si8aK4tlMTByUTXe7DSvZqyL+q+08gPFb534tXU7K7Om9OQWSGcnfx48ntnsphOSeqMrrXKKV3r9mfgocm6+SJkTFLYQHx2LiqqRhD6XGUHqa+zglbhfY2FuBr5JZX84BW4/14n9qxiudgmdfc6GV9r5Z057sac6Jkqs4Tr3Fd0rtNvrTUXSSCko1559zV51t0EgcVHhiOe+Ih7chMmf8T63mJdEAjen/EWZS+syw/B6I6AJCPYWARykOsj+a+vU1G3fMBSp08DOgIXKlrmIl004cS76Q6IoY/8qXLDkTsW9DBMo710qG+/845Jfe1tQE1rxyW0RyYVxo5ZCjg87LYfw8lVvVcQ+w4XpoSCDseUvRapM8MnxkbXLanZEXJGupd4rmuhloNYepW4ZOui8MqehDGHwFCje4RbeszF6CYGYjvgcF+a6AIikkzv+TOAaLUObvmVASM5U+wupTEkIXXRRtf2C41mvt8q5+KlUmcFm3YI1NHaRQlu8dKFgD/nBg5mVe8OCn+xG7JtGW7JrK0SaNN9EJrbzWCOB9ZeQFFf6gePaHDoj8o8cv8xUKZOINAY3qt5sTQwr0lpoe5bmco+jXJ3pzQohIZrbVdc7u2XOuSplu7KrZcoKZ/e4ominoyUyzJ3Kdev5+YLpGuHszrKQyQiQ+WLJJH+mS+YGEktddQbqPzbOI+8FFuIDLl9zsgGvpEjxY1CydKX+5xLWHBdZq1C/BrtNsm+u+IC09GoXI34wbeO6hN6DZeXIlFVjF3p3GbPM4nwzPRWh1NyGnRcAQS5Y+725alopcI3GZarC+CyLBmD/oC7XL37dPWBapf453T8iygfkcAlciDvZEQLom+0EivKu7jOM6ZqNmjZ9HtGY3feSvLlOfWD7Wt6gtMZ8TQKsq3z7vbTi8dbra+BoMn6azCC7oflaPvvBI76928ef9d58mzbg0u2GRD8JvbY6m5tPyczfuuIfsO92Pp3+gMZAqiwVB27HBN5psh8KjC8KYD0T1ahyp7CuK6cwdjq9vo7g929/YOnR98fvzj54cefTv/rry9fvX7z9uezd+9/+fW33//2d4Z5m4Deav3ri7akPRnvFuGabIPQo1LvVm5scBk/FgqyuvNL/5iLSrvSDUaknDj2DlThLMRAO1iB55v0uHQ48xhzI/VauuBD0Fm6RLyqORhy2aGBgTCOocX9a6B5OFuY6MFb16V5ttV6EP0q+eZFrf11mXb9pesGFV/sB/8iWl9x2DpsvjZ392v4CrZS8cAmAQz8JIDmJahHnRvW01VEmLkbZzVd1I0C/xbmtZNqksb2rsra0aCqGWFFqFHpzXYFQC+5lnYeLPD9XeDS9VOV2g9q3el4Tvq8aESIW+fzQA6b7/Mh4d7+xYrcBi1mbH5D2bva8EVRw9fYvDIZMw1f//Xo9X+drv4aztLqt78sNcLfTzH5EXgWdO96bdKrJtOMS67XosauMbUJOG2O0+VsY3/CrvEAjU022J5t7E/fNT4wjRkU29hHhmu8j8ZIrOYfauxyFSo1TerZCl5sz1qqLZ25kOi6xawD/WqquvajuEy9HPX6yaHxKpy7ZdK70OCt8n3szFM66kleXmOBxiN+w6/p6uxHZBjIIQVnMutWKMPNgoUrJyLHnicIk/EfizhDAoAUxxGXHFUeOMbX/+BtFQW4EUMfjcPXz5/TW+sOx/3aHK9jntPXxgemJ3NibdVRuKTFq2hrJ6CCpzmhGNfus/OlEK6dQ/QJyIi/JUkEmUQA3Xl5uD2ZpTl9decg4gmCfIQ4kOnyPk1fpsmVTNLNV9PIGdCgbxDmYMVCij6KGufbJhEDgq0Jm9qINWvE/4bYB4FEttV4quvjykVlXMxGqofQUNtktiGFJBdGXpUvNkFRkdsQW2VRaO+P15lQHXSjr9RbfYEuv66PV0lyCc6J2Prv+uymvm+PAGSgNjcURnbjc6IGD16vHNJRAeemhOOPqJ5okRTK7ZlSjTOFw4Ju6xkbw2gULgQRSBohYX7NBqyGMdd72Q0Dt4ABKVPOZ1exK+PVK49gtoLgeXconDofjCjUHgjKovFiFDV5c5mw15gRfcw1XN04rLddaoe+QRelpLmMUf0F7XSV1SgomE+6zZGUC88yK+12qSDqXHVK1nqLe3+bpfPwylxUL/Sjy/NqjPOaZ09wJbPwx5xYYPP/Az9el2s="


def _ensure_embedded_qrcodegen() -> None:
    global _QrCodeGenClass, _QrSegmentGenClass
    if _QrCodeGenClass is not None and _QrSegmentGenClass is not None:
        return
    namespace: Dict[str, Any] = {}
    try:
        source = zlib.decompress(b64decode(_QR_CODEGEN_B64)).decode("utf-8")
        exec(source, namespace)
        _QrCodeGenClass = namespace.get("QrCode")
        _QrSegmentGenClass = namespace.get("QrSegment")
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("Unable to load embedded QR generator: %s", exc)
        _QrCodeGenClass = None
        _QrSegmentGenClass = None


@dataclass
class Pk3ScaleEntry:
    name: str
    category: str
    original_width: int
    original_height: int
    new_width: int
    new_height: int
    path: str
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None

    @property
    def scale_x(self) -> float:
        if self.original_width <= 0 or self.new_width <= 0:
            return 1.0
        return self.new_width / self.original_width

    @property
    def scale_y(self) -> float:
        if self.original_height <= 0 or self.new_height <= 0:
            return 1.0
        return self.new_height / self.original_height


@dataclass
class DecodedGraphic:
    width: int
    height: int
    rgba: bytes
    left_offset: int = 0
    top_offset: int = 0
    has_offsets: bool = False


@dataclass
class TextureJob:
    input_path: Path
    output_path: Path
    detail_scale: int
    target_scale: int
    dry_run: bool
    identifier: str
    category: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CharacterModelerConfig:
    command: str
    weights: Path
    output_dir: Path
    mesh_format: str
    device: str
    max_angles: int
    fps: float


@dataclass
class CharacterFrameDescriptor:
    character: str
    animation: str
    frame_token: str
    angles: List[str]


class CharacterSpriteCollector:
    def __init__(self, sprite_root: Path) -> None:
        self.sprite_root = sprite_root
        self.sprite_root.mkdir(parents=True, exist_ok=True)
        self._characters: Dict[str, PixSpriteCharacter] = {}
        self._animation_lookup: Dict[str, Dict[str, PixSpriteAnimation]] = {}
        self._frame_counters: Dict[Tuple[str, str, str], int] = defaultdict(int)

    def add_frame(self, identifier: str, image_bytes: bytes, extension: Optional[str]) -> None:
        if not CHARACTER_MODELER_AVAILABLE or PixSpriteCharacter is None or PixSpriteFrame is None:
            return
        descriptor = _parse_character_frame_descriptor(identifier)
        if not descriptor:
            return
        ext = extension or ".png"
        if not ext.startswith("."):
            ext = f".{ext}"
        character_dir = self.sprite_root / sanitize_pix2vox_name(descriptor.character)
        animation_dir = character_dir / sanitize_pix2vox_name(descriptor.animation)
        animation_dir.mkdir(parents=True, exist_ok=True)
        frame_name = f"{sanitize_pix2vox_name(descriptor.frame_token)}{ext.lower()}"
        frame_path = animation_dir / frame_name
        try:
            frame_path.write_bytes(image_bytes)
        except Exception as exc:
            logging.debug("Failed to stage sprite %s for Pix2Vox+: %s", identifier, exc)
            return
        try:
            rel_path = frame_path.relative_to(self.sprite_root)
        except ValueError:
            rel_path = frame_path
        animation = self._ensure_animation(descriptor.character, descriptor.animation)
        for angle in descriptor.angles:
            counter_key = (descriptor.character, descriptor.animation, angle)
            frame_index = self._frame_counters[counter_key]
            self._frame_counters[counter_key] = frame_index + 1
            frame = PixSpriteFrame(
                path=frame_path,
                rel_path=rel_path,
                angle=angle,
                frame_index=frame_index,
                animation=descriptor.animation,
            )
            animation.frames_by_angle.setdefault(angle, []).append(frame)

    def _ensure_animation(self, character_name: str, animation_name: str) -> PixSpriteAnimation:
        assert PixSpriteCharacter is not None and PixSpriteAnimation is not None
        character = self._characters.get(character_name)
        if character is None:
            character = PixSpriteCharacter(name=character_name, animations=[])
            self._characters[character_name] = character
            self._animation_lookup[character_name] = {}
        animation_map = self._animation_lookup[character_name]
        animation = animation_map.get(animation_name)
        if animation is None:
            animation = PixSpriteAnimation(name=animation_name)
            animation_map[animation_name] = animation
            character.animations.append(animation)
        return animation

    def has_characters(self) -> bool:
        return bool(self._characters)

    def iter_characters(self) -> List[PixSpriteCharacter]:
        return list(self._characters.values())


def _parse_character_frame_descriptor(identifier: str) -> Optional[CharacterFrameDescriptor]:
    base = Path(identifier).stem
    if not base:
        return None
    match = CHARACTER_FRAME_PATTERN.search(base.lower())
    if not match:
        return None
    prefix = base[: match.start()].strip("_- ")
    if not prefix:
        return None
    suffix = base[match.start() :].upper()
    angles: List[str] = []
    animation_letter: Optional[str] = None
    index = 0
    while index + 1 < len(suffix):
        letter = suffix[index]
        digit = suffix[index + 1]
        if not letter.isalpha() or not digit.isdigit():
            break
        token = f"{letter}{digit}"
        angles.append(token)
        if animation_letter is None:
            animation_letter = letter
        index += 2
    if not angles or animation_letter is None:
        return None
    animation_name = f"frame_{animation_letter.upper()}"
    character_name = prefix.upper()
    return CharacterFrameDescriptor(
        character=character_name,
        animation=animation_name,
        frame_token=suffix,
        angles=angles,
    )


def run_character_modeler_pipeline(
    collector: CharacterSpriteCollector,
    config: CharacterModelerConfig,
    *,
    dry_run: bool,
) -> None:
    if not CHARACTER_MODELER_AVAILABLE or PixSpriteCharacter is None:
        logging.error("Pix2Vox+ integration requested but character_modeler.py is unavailable.")
        return
    characters = collector.iter_characters()
    if not characters:
        return
    ensure_pix2vox_output_dir(config.output_dir)
    logging.info("Generating Pix2Vox+ meshes for %d character(s)", len(characters))
    for character in characters:
        character_dir = config.output_dir / sanitize_pix2vox_name(character.name)
        ensure_pix2vox_output_dir(character_dir)
        mesh_path = character_dir / f"{sanitize_pix2vox_name(character.name)}.{config.mesh_format}"
        views = prepare_pix2vox_views(character, config.max_angles)
        if not views:
            logging.warning("Character %s does not expose distinct viewing angles; skipping Pix2Vox+.", character.name)
            continue
        with stage_views_for_pix2vox(views) as staging_dir:
            replacements = {
                "input": staging_dir,
                "output": str(mesh_path),
                "weights": str(config.weights),
                "format": config.mesh_format,
                "device": config.device,
                "name": character.name,
            }
            try:
                run_pix2vox_command(config.command, replacements, dry_run=dry_run)
            except Pix2VoxCommandError as exc:
                logging.error("Pix2Vox+ failed for %s: %s", character.name, exc)
                continue
        for animation in character.animations:
            write_keyframe_file(character_dir, animation, fps=config.fps)
        write_character_metadata(character_dir, character, views, mesh_path, collector.sprite_root)

def record_arcade_score(value: int) -> None:
    score = max(0, int(value))
    ARCADE_HIGH_SCORES.append(score)
    ARCADE_HIGH_SCORES.sort(reverse=True)
    del ARCADE_HIGH_SCORES[ARCADE_MAX_HIGH_SCORES:]
    save_arcade_scores(ARCADE_HIGH_SCORES)



SPRITE_KEYWORDS = (
    "sprite",
    "sprites/",
    "decor",
    "weapon",
    "weapons/",
)
CHARACTER_KEYWORDS = (
    "actor",
    "actors/",
    "character",
    "characters/",
    "monster",
    "monsters/",
    "enemy",
    "enemies",
    "npc",
    "player",
    "players/",
    "marine",
    "boss",
    "death",
    "dead",
    "corpse",
    "remains",
    "gib",
    "gibs",
    "gore",
)
EXTRA_CHARACTER_KEYWORDS: set[str] = set()
CHARACTER_FRAME_PATTERN = re.compile(r"[a-z][0-9](?:[a-z][0-9])?$")
NORMAL_KEYWORDS = (
    "_n.",
    "_nrm",
    "_normal",
    "_norm",
    "_normalmap",
    "-n.",
    " normal",
)
MASK_KEYWORDS = (
    "_mask",
    "_opacity",
    "_alpha",
    "_trans",
    "_rough",
    "_metal",
    "_ao",
    "_spec",
)
UI_KEYWORDS = (
    "hud",
    "hud/",
    "/hud",
    "ui/",
    "/ui",
    "ui_",
    "menu",
    "mnu",
    "font",
    "status",
    "stbar",
    "title",
    "intermission",
    "credit",
    "credits",
    "mugshot",
    "help",
    "cursor",
    "dialog",
    "console",
    "crosshair",
    "reticle",
    "icon",
    "option",
)

def _edt_1d(f: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    n = f.shape[0]
    result = np.empty_like(f)
    v = np.zeros(n, dtype=int)
    z = np.zeros(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -math.inf
    z[1] = math.inf
    for q in range(1, n):
        while True:
            r = v[k]
            s = ((f[q] + q * q) - (f[r] + r * r)) / (2 * q - 2 * r)
            if s <= z[k]:
                k -= 1
                if k < 0:
                    k = 0
                    break
            else:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = math.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        r = v[k]
        val = (q - r) * (q - r) + f[r]
        result[q] = val
    return result


def _distance_transform_boolean(mask: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    f = np.where(mask, 0.0, np.inf).astype(np.float64)
    if f.ndim != 2:
        raise ValueError("Distance transform expects a 2D array")
    rows, cols = f.shape
    for x in range(cols):
        f[:, x] = _edt_1d(f[:, x])
    for y in range(rows):
        f[y, :] = _edt_1d(f[y, :])
    with np.errstate(invalid="ignore"):
        f = np.sqrt(f)
    f[np.isinf(f)] = 0.0
    return f



DEFAULT_WORLD_PROMPT = (
    "high quality detailed game texture with micro surface detail, realistic shading, consistent scale, seamless, game-ready, muted color palette"
)
DEFAULT_WORLD_NEGATIVE = (
    "blurry, oversmoothed, macro swirls, abstract shapes, watercolor wash, artifacts, watermark, text, neon colors"
)

SPRITE_PROMPT = (
    "high quality game sprite, crisp silhouettes, consistent lighting, preserved proportions, clean shading, game-ready, natural color palette"
)
SPRITE_NEGATIVE = (
    "blurry edges, warped anatomy, extreme deformations, painterly smears, text overlay, artifacts, neon outlines"
)

UI_PROMPT = (
    "sharp pixel-perfect UI element, high contrast glyphs, clean edges, game interface icon, consistent style, restrained colors"
)
UI_NEGATIVE = (
    "blurry, distorted lettering, heavy gradients, drop shadows, macro swirls, random shapes, artifacts, neon glow"
)


def _apply_unsharp_mask_image(image: "Image.Image", strength: str = "default") -> "Image.Image":
    try:
        from PIL import ImageFilter  # type: ignore
    except ImportError:
        return image

    presets = {
        "default": (0.6, 110, 1),
        "strong": (0.62, 115, 1),
    }
    radius, percent, threshold = presets.get(strength, presets["default"])

    if image.mode == "RGBA":
        rgb = image.convert("RGB")
        alpha = image.split()[3]
        sharpened_rgb = rgb.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        result = sharpened_rgb.convert("RGBA")
        result.putalpha(alpha)
        return result

    if image.mode not in {"RGB", "L"}:
        working = image.convert("RGB")
        sharpened = working.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        return sharpened.convert(image.mode)
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def _apply_unsharp_mask_file(path: Path, strength: str = "default") -> None:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return
    try:
        with Image.open(path) as img:
            sharpened = _apply_unsharp_mask_image(img, strength)
            sharpened.save(path)
    except Exception as exc:
        logging.debug("Post-sharpen skipped for %s: %s", path, exc)


def _is_character_keyword_match(name: str) -> bool:
    for token in CHARACTER_KEYWORDS:
        if token in name:
            return True
    for token in EXTRA_CHARACTER_KEYWORDS:
        if token in name:
            return True
    return False


def _looks_like_character_frame(base: str) -> bool:
    if not base:
        return False
    return bool(CHARACTER_FRAME_PATTERN.search(base))


def detect_texture_category(identifier: str, source_type: Optional[str] = None) -> str:
    name = identifier.replace("\\", "/").lower()
    base = Path(name).stem
    if base.endswith(("_n", "_nm", "_nrm", "_norm", "_normal", "_normalmap")):
        return "normal"
    if base.endswith(("_mask", "_opacity", "_alpha", "_rough", "_ao", "_metal", "_metallic", "_spec")):
        return "mask"
    if source_type == "sprite":
        if _is_character_keyword_match(name) or _looks_like_character_frame(base.lower()):
            return "character"
        return "sprite"
    if source_type == "patch":
        if name.startswith("m_") or base.startswith("m_"):
            return "ui"
        if any(token in name for token in UI_KEYWORDS):
            return "ui"
        return "world"
    if source_type == "flat":
        return "world"

    if any(token in name for token in NORMAL_KEYWORDS):
        return "normal"
    if any(token in name for token in MASK_KEYWORDS):
        return "mask"
    if _is_character_keyword_match(name) or _looks_like_character_frame(base.lower()):
        return "character"
    if any(token in name for token in UI_KEYWORDS):
        return "ui"
    if any(token in name for token in SPRITE_KEYWORDS):
        return "sprite"
    return "world"



def _generate_hires_patch_name(base: str, existing: set[str]) -> str:
    cleaned = sanitize_lump_name(base).upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not cleaned:
        cleaned = "HD"
    prefix = cleaned[:7] if len(cleaned) > 1 else cleaned
    candidate = (prefix + "H")[:8] or "HD"
    counter = 0
    while candidate in existing or not candidate:
        suffix = f"H{counter}"
        prefix = cleaned[: max(0, 8 - len(suffix))]
        if not prefix:
            prefix = "H"
        candidate = (prefix + suffix)[:8]
        counter += 1
    existing.add(candidate)
    return candidate


def _format_texture_token(name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        return name
    return f"\"{name}\""


def _compute_normal_map(image: np.ndarray, strength: float = 2.0) -> np.ndarray:
    alpha_channel: Optional[np.ndarray] = None
    if image.ndim == 2:
        height_map = image.astype(np.float32) / 255.0
    else:
        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3].astype(np.uint8)
            color = image[:, :, :3]
        else:
            color = image[:, :, :3]
        height_map = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    height_map = cv2.GaussianBlur(height_map, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0)
    sobel_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
    normal_x = -sobel_x * strength
    normal_y = -sobel_y * strength
    normal_z = np.ones_like(height_map)
    normal = np.stack((normal_x, normal_y, normal_z), axis=-1)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1.0
    normal /= norm
    normal = (normal * 0.5 + 0.5) * 255.0
    normal = np.clip(normal, 0, 255).astype(np.uint8)
    normal = normal[:, :, [2, 1, 0]]  # Put Z into blue for OpenCV's BGR encoding
    if alpha_channel is not None:
        normal = np.dstack((normal, alpha_channel))
    return normal


def _generate_normal_map_bytes(image_bytes: bytes, strength: float = 2.0) -> Optional[bytes]:
    if cv2 is None:
        logging.debug("OpenCV is not available; skipping normal map generation.")
        return None
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    if buffer.size == 0:
        return None
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        logging.debug("Failed to decode image bytes for normal map generation.")
        return None
    normal = _compute_normal_map(decoded, strength=strength)
    success, encoded = cv2.imencode(".png", normal)
    if not success:
        logging.debug("Failed to encode normal map as PNG.")
        return None
    return encoded.tobytes()


def _build_normal_map_path(original: str) -> str:
    path = PurePosixPath(original.replace("\\", "/"))
    normal_name = f"{path.stem}_NM.png"
    return str(PurePosixPath("normal") / normal_name)


def _generate_specular_map_bytes(image_bytes: bytes) -> Optional[bytes]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            specular_image = ImageOps.grayscale(image)
            specular_image = ImageEnhance.Brightness(specular_image).enhance(3.0)
            specular_image = ImageEnhance.Contrast(specular_image).enhance(3.0)
            width, height = specular_image.size
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    buffer = io.BytesIO()
    specular_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_specular_map_path(original: str) -> str:
    path = PurePosixPath(original.replace("\\", "/"))
    specular_name = f"{path.stem}_SP.png"
    return str(PurePosixPath("specular") / specular_name)


def _generate_normal_lump_name(identifier: str, existing: set[str]) -> str:
    base = sanitize_lump_name(identifier).upper()
    base = re.sub(r"[^A-Z0-9]", "", base)
    if not base:
        base = "TEX"
    suffix = "_NM"
    prefix_max = max(0, 8 - len(suffix))
    prefix_base = base[:prefix_max] if prefix_max else ""
    if not prefix_base:
        prefix_base = "T"
    counter = 0
    while True:
        if counter == 0:
            prefix = prefix_base
        else:
            counter_str = str(counter)
            prefix = prefix_base[: max(0, prefix_max - len(counter_str))] + counter_str
            if not prefix:
                prefix = counter_str[-prefix_max:]
        candidate = f"{prefix}{suffix}"
        if len(candidate) > 8:
            counter += 1
            continue
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        counter += 1


def _generate_specular_lump_name(identifier: str, existing: set[str]) -> str:
    base = sanitize_lump_name(identifier).upper()
    base = re.sub(r"[^A-Z0-9]", "", base)
    if not base:
        base = "TEX"
    suffix = "_SP"
    prefix_max = max(0, 8 - len(suffix))
    prefix_base = base[:prefix_max] if prefix_max else ""
    if not prefix_base:
        prefix_base = "T"
    counter = 0
    while True:
        if counter == 0:
            prefix = prefix_base
        else:
            counter_str = str(counter)
            prefix = prefix_base[: max(0, prefix_max - len(counter_str))] + counter_str
            if not prefix:
                prefix = counter_str[-prefix_max:]
        candidate = f"{prefix}{suffix}"
        if len(candidate) > 8:
            counter += 1
            continue
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        counter += 1


def _extract_material_map_references(
    data: bytes,
) -> Tuple[Set[str], Set[str], Dict[Tuple[str, str], Dict[str, str]]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\r\n]*", "", text)
    path_refs: Set[str] = set()
    name_refs: Set[str] = set()
    material_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    material_pattern = re.compile(
        r"\bmaterial\s+(\w+)\s+((?:\"[^\"]+\"|'[^']+'|[^\s\{]+))\s*\{",
        re.IGNORECASE,
    )
    pattern = re.compile(
        r"\b(normal|specular)\b\s+((?:\"[^\"]+\"|'[^']+'|[^\s\{\};]+))",
        re.IGNORECASE,
    )
    length = len(text)

    def _strip_token(token: str) -> str:
        cleaned = token.strip().rstrip(",;")
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
            cleaned = cleaned[1:-1]
        return cleaned.strip()

    def _register_reference(token: str) -> None:
        normalized = token.replace("\\", "/")
        path_refs.add(normalized.lower())
        if "/" not in normalized:
            name_refs.add(normalized.upper())
        else:
            stem = PurePosixPath(normalized).stem
            if stem:
                name_refs.add(stem.upper())

    def _normalize_target(material_type: str, target_token: str) -> Tuple[str, str]:
        material_type_lower = material_type.lower()
        if material_type_lower in {"sprite", "flat"}:
            normalized_target = sanitize_lump_name(target_token).upper()
        else:
            path_token = target_token.replace("\\", "/")
            path_obj = PurePosixPath(path_token)
            if path_obj.suffix:
                try:
                    path_obj = path_obj.with_suffix("")
                except ValueError:
                    pass
            normalized_target = str(path_obj).lower()
        return material_type_lower, normalized_target

    idx = 0
    while True:
        match = material_pattern.search(text, idx)
        if not match:
            break
        material_type = match.group(1)
        target_raw = match.group(2)
        target_token = _strip_token(target_raw)
        brace_pos = match.end()
        depth = 1
        pos = brace_pos
        while pos < length and depth > 0:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            pos += 1
        block = text[brace_pos : pos - 1 if depth == 0 else length]
        idx = pos
        key = _normalize_target(material_type, target_token)
        refs = material_map.setdefault(key, {})
        for ref_match in pattern.finditer(block):
            ref_kind = ref_match.group(1).lower()
            raw_token = ref_match.group(2)
            token = _strip_token(raw_token)
            if not token:
                continue
            normalized_token = token.replace("\\", "/")
            _register_reference(normalized_token)
            refs[ref_kind] = normalized_token

    # Fallback: capture stray normal/specular directives outside material blocks.
    if not material_map:
        for ref_match in pattern.finditer(text):
            raw_token = ref_match.group(2)
            token = _strip_token(raw_token)
            if not token:
                continue
            normalized_token = token.replace("\\", "/")
            _register_reference(normalized_token)

    return path_refs, name_refs, material_map


def _build_material_block(materials: Dict[Tuple[str, str], Dict[str, str]]) -> Optional[str]:
    if not materials:
        return None
    lines: List[str] = ["// Auto-generated by texture_upscaler normal map materials", ""]
    for (material_type, target) in sorted(materials.keys(), key=lambda item: (item[0], item[1])):
        record = materials[(material_type, target)]
        normal_ref = record.get("normal")
        if not normal_ref:
            continue
        lines.append(f"material {material_type} {_format_texture_token(target)}")
        lines.append("{")
        lines.append(f"    normal {_format_texture_token(normal_ref)}")
        specular_ref = record.get("specular")
        if specular_ref:
            lines.append(f"    specular {_format_texture_token(specular_ref)}")
        lines.append(f"    specularlevel {format_float(SPECULAR_LEVEL)}")
        lines.append("}")
        lines.append("")
    if len(lines) <= 2:
        return None
    return ("\n".join(lines)).rstrip() + "\n"


def build_hirestex_payload(
    entries: Iterable[Pk3ScaleEntry],
    allowed_categories: Optional[Collection[str]] = None,
    include_unscaled: bool = False,
) -> Optional[bytes]:
    header = "// Auto-generated by texture_upscaler scale overrides"
    lines: List[str] = [header, ""]
    unique_entries: Dict[str, Pk3ScaleEntry] = {}
    allowed_set = set(allowed_categories) if allowed_categories is not None else None
    for entry in entries:
        if allowed_set is not None and entry.category not in allowed_set:
            continue
        if entry.new_width <= 0 or entry.new_height <= 0:
            continue
        scale_x = entry.scale_x
        scale_y = entry.scale_y
        if not include_unscaled and abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
            continue
        unique_entries.setdefault(entry.name, entry)
    if not unique_entries:
        return None
    for entry in sorted(unique_entries.values(), key=lambda item: item.name):
        if entry.category in {"sprite", "character"}:
            kind = "sprite"
        elif entry.category == "world":
            kind = "texture"
        elif entry.category == "ui":
            kind = "graphic"
        else:
            kind = "graphic"
        if entry.new_width <= 0 or entry.new_height <= 0:
            continue
        scale_x = entry.scale_x
        scale_y = entry.scale_y
        if abs(scale_x) < 1e-9 or abs(scale_y) < 1e-9:
            continue
        xscale_value = scale_x if entry.original_width else 1.0
        yscale_value = scale_y if entry.original_height else 1.0
        lines.append(f"{kind} {_format_texture_token(entry.name)}, {entry.new_width}, {entry.new_height}")
        lines.append("{")
        lines.append(f"    Patch \"{entry.path}\", 0, 0")
        if entry.category in {"sprite", "character", "ui"}:
            if entry.offset_x is not None and entry.offset_y is not None:
                lines.append(f"    Offset {int(entry.offset_x)}, {int(entry.offset_y)}")
            else:
                lines.append("    {")
                lines.append("        UseOffsets")
                lines.append("    }")
        lines.append(f"    XScale {format_float(xscale_value)}")
        lines.append(f"    YScale {format_float(yscale_value)}")
        if entry.category == "world" and (
            abs(xscale_value - 1.0) > 1e-6 or abs(yscale_value - 1.0) > 1e-6
        ):
            lines.append("    WorldPanning")
        lines.append("}")
        lines.append("")
    if len(lines) <= 2:
        return None
    hires_data = ("\n".join(lines)).rstrip() + "\n"
    return hires_data.encode("utf-8")


def _determine_material_kind_pk3(category: str, arcname: str) -> str:
    normalized = arcname.replace("\\", "/").lower()
    if category in {"sprite", "character"} or normalized.startswith("sprites/"):
        return "sprite"
    if normalized.startswith("flats/") or "/flats/" in normalized:
        return "flat"
    return "texture"


def _material_target_from_pk3(material_type: str, arcname: str) -> str:
    path = PurePosixPath(arcname.replace("\\", "/"))
    if material_type in {"sprite", "flat"}:
        return sanitize_lump_name(path.stem).upper()
    return str(path.with_suffix(""))


def _determine_material_kind_wad(source_type: str, category: str) -> str:
    if source_type == "flat":
        return "flat"
    if source_type == "sprite":
        return "sprite"
    return "texture"


def _material_target_from_wad(lump_name: str) -> str:
    return sanitize_lump_name(lump_name).upper()


def build_wad_diff_arcname(lump_name: str, category: str, source_type: str, suffix: str = ".png") -> str:
    sanitized = sanitize_lump_name(lump_name).upper()
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    directory = "textures"
    if source_type == "flat":
        directory = "flats"
    elif source_type == "sprite" or category in {"sprite", "character"}:
        directory = "sprites"
    elif category == "ui":
        directory = "graphics"
    arcname = PurePosixPath(directory) / f"{sanitized}{ext.lower()}"
    return str(arcname)


def _precount_pk3_textures(
    archive: zipfile.ZipFile,
    infos: Sequence[zipfile.ZipInfo],
    normalized_exts: set[str],
    use_all: bool,
    max_pixels: int,
    skip_categories: Collection[str] = (),
    locked_paths: Collection[str] = (),
    locked_names: Collection[str] = (),
) -> int:
    total = 0
    skip_category_set = {category.lower() for category in skip_categories if category}
    locked_path_set = {path.lower() for path in locked_paths if path}
    locked_name_set = {name.upper() for name in locked_names if name}
    for info in infos:
        arcname = info.filename
        if arcname.endswith("/"):
            continue
        try:
            data = archive.read(arcname)
        except KeyError:
            continue
        suffix = Path(arcname).suffix.lower()
        detected_ext = detect_image_extension(data)
        allowed_extension: Optional[str] = None
        if suffix and (use_all or suffix in normalized_exts):
            allowed_extension = suffix
        elif detected_ext and (use_all or detected_ext in normalized_exts):
            allowed_extension = detected_ext
        if allowed_extension is None:
            continue
        arcname_posix = arcname.replace("\\", "/")
        arcname_lower = arcname_posix.lower()
        if locked_path_set and arcname_lower in locked_path_set:
            continue
        if locked_name_set:
            stem_token = PurePosixPath(arcname_posix).stem.upper()
            if stem_token in locked_name_set:
                continue
        category = detect_texture_category(arcname)
        if category in {"normal", "mask"}:
            continue
        if skip_category_set and category in skip_category_set:
            continue
        total += 1
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract textures from a PK3 (zip) or WAD archive, upscale them with the NMKD Yandere "
            "Pixelart RealESRGAN model (PyTorch or NCNN), and rebuild an enhanced archive."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to the .pk3 or .wad file to enhance.",
    )
    parser.add_argument(
        "--realesrgan-bin",
        default=None,
        help="Path to the realesrgan-ncnn-vulkan executable. Defaults to auto-detect in the current installation.",
    )
    parser.add_argument(
        "--enable-sprite-materials",
        action="store_true",
        help=(
            "Generate normal/specular maps for sprite, character, and UI assets. "
            "Disabled by default to avoid over-processing first-person weapons and HUD graphics."
        ),
    )
    parser.add_argument(
        "--disable-materials",
        action="store_true",
        help="Skip writing any normal or specular material definitions for generated textures.",
    )
    parser.add_argument(
        "--realesrgan-pth",
        type=Path,
        help=(
            "Path to a RealESRGAN .pth model (PyTorch). When provided, the library backend is used. "
            "Defaults to auto-detecting 8x_NMKD-YanderePixelart4_150000_G.pth in the project."
        ),
    )
    parser.add_argument(
        "--realesrgan-gpu",
        default="auto",
        help="GPU index passed to realesrgan-ncnn-vulkan (-g). Use 'auto' for the driver's default.",
    )
    parser.add_argument(
        "--realesrgan-tile",
        type=int,
        default=0,
        help="Optional tile size passed to realesrgan-ncnn-vulkan (0 disables tiling).",
    )
    parser.add_argument(
        "--realesrgan-tile-pad",
        type=int,
        default=10,
        help="Tile padding passed to realesrgan-ncnn-vulkan.",
    )
    parser.add_argument(
        "--enable-post-sharpen",
        action="store_true",
        default=True,
        help="Apply a subtle unsharp mask after blending (enabled by default).",
    )

    parser.add_argument(
        "--esrgan-strength",
        type=float,
        default=0.5,
        help=(
            "Blend factor for ESRGAN detail versus the original bicubic upscale "
            "(1.0 keeps pure ESRGAN, lower values tame high-contrast artifacts)."
        ),
    )
    parser.add_argument(
        "--esrgan-detail-limit",
        type=float,
        default=0.0,
        help=(
            "Clamp per-channel ESRGAN detail (0 disables clamping). "
            "Lower values reduce ringing/noise along edges."
        ),
    )
    parser.add_argument(
        "--detail-scale",
        type=int,
        default=None,
        help=(
            "Intermediate upscale factor sent to the AI upscaler before the bicubic downscale "
            "step. Defaults to the model scale when using --realesrgan-pth; otherwise 8."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help=(
            "Final output scale after bicubic downscaling. "
            "The AI upscaler runs at --detail-scale first, then results are resized to this factor. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=300_000,
        help=(
            "Skip upscaling images whose width*height exceeds this many pixels. "
            "Set to 0 to disable the limit. Default: 300000."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where the enhanced archive will be written. Defaults to the source directory.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("diff", "full"),
        default="diff",
        help=(
            "Select the output archive layout. "
            "'diff' writes a PK3 containing only the upscaled assets (default). "
            "'full' rewrites the entire source archive with embedded changes."
        ),
    )
    parser.add_argument(
        "--full-output",
        dest="output_mode",
        action="store_const",
        const="full",
        help="Legacy alias for --output-mode full.",
    )
    parser.add_argument(
        "--texture-extensions",
        default=".png,.jpg,.jpeg,.tga,.dds,.gif,.bmp,.webp,.tiff",
        help=(
            "Comma-separated list of texture file extensions to process inside PK3 archives. "
            "Case-insensitive. Use '*' to process every file. "
            "Default: .png,.jpg,.jpeg,.tga,.dds,.gif,.bmp,.webp,.tiff"
        ),
    )
    parser.add_argument(
        "--character-keywords",
        metavar="LIST",
        action="append",
        default=[],
        help=(
            "Comma-separated list of extra keywords that should classify textures as characters. "
            "Useful for WAD sprite lumps (e.g., PLAYA1) or custom folder names."
        ),
    )
    parser.add_argument(
        "--pix2vox-command",
        help=(
            "Command template that invokes Pix2Vox+. Use placeholders {input}, {output}, {weights}, {format}, {device}, and {name}."
        ),
    )
    parser.add_argument(
        "--pix2vox-weights",
        type=Path,
        help="Path to the Pix2Vox+ checkpoint (.pth) that the command expects.",
    )
    parser.add_argument(
        "--pix2vox-mesh-format",
        choices=("obj", "ply", "gltf", "glb"),
        default="obj",
        help="Mesh format requested from Pix2Vox+. Default: obj.",
    )
    parser.add_argument(
        "--pix2vox-device",
        default="cuda",
        help="Device identifier forwarded to the Pix2Vox+ command template (default: cuda).",
    )
    parser.add_argument(
        "--pix2vox-max-angles",
        type=int,
        default=24,
        help="Maximum number of unique sprite angles forwarded to Pix2Vox+.",
    )
    parser.add_argument(
        "--pix2vox-fps",
        type=float,
        default=15.0,
        help="Frame rate recorded in the generated animation keyframes (default: 15).",
    )
    parser.add_argument(
        "--character-output-dir",
        type=Path,
        default=Path("characters"),
        help="Directory that receives generated Pix2Vox+ character meshes and metadata.",
    )
    parser.add_argument(
        "--skip-types",
        "--skip-categories",
        dest="skip_categories",
        metavar="LIST",
        action="append",
        default=[],
        help=(
            "Comma-separated list of texture categories to skip during upscaling. "
            f"Supported categories: {', '.join(TEXTURE_CATEGORY_CHOICES)}."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary working directory instead of deleting it after completion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover textures and show the commands that would run without modifying anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def create_temp_dir(keep: bool) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="texture_upscaler_"))
    if keep:
        logging.info("Temporary workspace: %s", temp_root)
    return temp_root


def cleanup_temp_dir(temp_root: Path, keep: bool) -> None:
    if keep:
        logging.info("Keeping temporary workspace at %s", temp_root)
        return
    shutil.rmtree(temp_root, ignore_errors=True)


def detect_archive_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pk3":
        return "pk3"
    if ext == ".wad":
        return "wad"
    raise ValueError(f"Unsupported archive type for {path}. Expected .pk3 or .wad.")


def build_enhanced_name(path: Path, suffix: Optional[str] = None) -> Path:
    target_dir = path.parent
    resolved_suffix = suffix if suffix is not None else path.suffix
    if resolved_suffix and not resolved_suffix.startswith("."):
        resolved_suffix = f".{resolved_suffix}"
    return target_dir / f"{path.stem}-enhanced{resolved_suffix}"


def sanitize_lump_name(name: str) -> str:
    if not name:
        return "unnamed"
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)


def parse_hirestex_entries(data: bytes) -> Dict[str, Pk3ScaleEntry]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    lines = text.splitlines()
    entries: Dict[str, Pk3ScaleEntry] = {}
    header_re = re.compile(
        r"^(sprite|graphic|texture)\s+([^,\s]+)\s*,\s*(\d+)\s*,\s*(\d+)",
        re.IGNORECASE,
    )
    offset_re = re.compile(r"offset\s+(-?\d+)\s*,\s*(-?\d+)", re.IGNORECASE)
    scale_re = re.compile(
        r"(xscale|yscale)\s+([+-]?(?:\d+\.\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    patch_re = re.compile(r"patch\s+(\"[^\"]+\"|[^,\s]+)", re.IGNORECASE)
    i = 0
    length = len(lines)
    while i < length:
        raw_line = lines[i].strip()
        i += 1
        if not raw_line or raw_line.startswith("//"):
            continue
        match = header_re.match(raw_line)
        if not match:
            continue
        kind = match.group(1).lower()
        name_token = match.group(2).strip()
        if name_token.startswith("\"") and name_token.endswith("\"") and len(name_token) >= 2:
            name_token = name_token[1:-1]
        sanitized_name = sanitize_lump_name(name_token).upper()
        new_width = int(match.group(3))
        new_height = int(match.group(4))
        while i < length and lines[i].strip() == "":
            i += 1
        if i >= length or lines[i].strip() != "{":
            continue
        i += 1
        path_token: Optional[str] = None
        offset_x: Optional[int] = None
        offset_y: Optional[int] = None
        xscale = 1.0
        yscale = 1.0
        while i < length:
            inner = lines[i].strip()
            i += 1
            if not inner or inner.startswith("//"):
                continue
            if inner == "}":
                break
            lower = inner.lower()
            if lower.startswith("patch"):
                patch_match = patch_re.match(inner)
                if patch_match:
                    token = patch_match.group(1).strip()
                    if token.startswith("\"") and token.endswith("\"") and len(token) >= 2:
                        token = token[1:-1]
                    path_token = token.replace("\\", "/")
            elif lower.startswith("offset"):
                offset_match = offset_re.search(inner)
                if offset_match:
                    offset_x = int(offset_match.group(1))
                    offset_y = int(offset_match.group(2))
            elif lower.startswith("xscale") or lower.startswith("yscale"):
                scale_match = scale_re.search(inner)
                if scale_match:
                    value = float(scale_match.group(2))
                    if scale_match.group(1).lower() == "xscale":
                        xscale = value
                    else:
                        yscale = value
            elif lower.startswith("useoffsets"):
                offset_x = None
                offset_y = None
        path = (path_token or sanitized_name).replace("\\", "/")
        category = "world"
        if kind == "sprite":
            category = "sprite"
        elif kind == "graphic":
            category = "ui"
        original_width = new_width
        original_height = new_height
        if abs(xscale) > 1e-6:
            original_width = int(round(new_width / xscale))
        if abs(yscale) > 1e-6:
            original_height = int(round(new_height / yscale))
        entry = Pk3ScaleEntry(
            name=sanitized_name,
            category=category,
            original_width=original_width,
            original_height=original_height,
            new_width=new_width,
            new_height=new_height,
            path=path,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        entries[sanitized_name] = entry
    return entries


def detect_image_extension(blob: bytes) -> Optional[str]:
    if blob.startswith(PNG_MAGIC):
        return ".png"
    if blob.startswith(JPEG_SOI):
        return ".jpg"
    if blob.startswith(BMP_MAGIC):
        return ".bmp"
    if blob.startswith(GIF87A) or blob.startswith(GIF89A):
        return ".gif"
    if blob.startswith(DDS_MAGIC):
        return ".dds"
    if len(blob) >= 12 and blob[:4] == RIFF_MAGIC and blob[8:12] == WEBP_MAGIC:
        return ".webp"
    if len(blob) >= 26 and TGA_FOOTER in blob[-26:]:
        return ".tga"
    # Basic TIFF check (endian markers)
    if blob.startswith(b"II*\x00") or blob.startswith(b"MM\x00*"):
        return ".tiff"
    return None


def extract_png_dimensions(data: bytes) -> Tuple[int, int]:
    if not data.startswith(PNG_MAGIC):
        return 0, 0
    idx = 8
    while idx + 8 <= len(data):
        if idx + 8 > len(data):
            break
        length = struct.unpack(">I", data[idx : idx + 4])[0]
        chunk_type = data[idx + 4 : idx + 8]
        chunk_end = idx + 8 + length
        if chunk_end + 4 > len(data):
            break
        chunk_data = data[idx + 8 : chunk_end]
        if chunk_type == b"IHDR" and len(chunk_data) >= 8:
            width, height = struct.unpack(">II", chunk_data[:8])
            return int(width), int(height)
        idx = chunk_end + 4
    return 0, 0


def extract_png_grab(data: bytes) -> Optional[Tuple[int, int]]:
    if not data.startswith(PNG_MAGIC):
        return None
    idx = 8
    while idx + 8 <= len(data):
        length = struct.unpack(">I", data[idx : idx + 4])[0]
        chunk_type = data[idx + 4 : idx + 8]
        chunk_end = idx + 8 + length
        if chunk_end + 4 > len(data):
            break
        chunk_data = data[idx + 8 : chunk_end]
        if chunk_type == b"grAb" and len(chunk_data) >= 8:
            return struct.unpack(">ii", chunk_data[:8])
        idx = chunk_end + 4
    return None


def set_png_grab_chunk(data: bytes, offsets: Tuple[int, int]) -> bytes:
    if not data.startswith(PNG_MAGIC):
        return data
    x_off, y_off = offsets
    grab_data = struct.pack(">ii", int(x_off), int(y_off))
    result = bytearray()
    result.extend(data[:8])
    idx = 8
    inserted = False
    while idx + 8 <= len(data):
        length = struct.unpack(">I", data[idx : idx + 4])[0]
        chunk_type = data[idx + 4 : idx + 8]
        chunk_end = idx + 8 + length
        chunk_crc_end = chunk_end + 4
        if chunk_end > len(data) or chunk_crc_end > len(data):
            break
        if chunk_type == b"grAb":
            # Skip existing grAb chunk
            idx = chunk_crc_end
            continue
        result.extend(data[idx:chunk_crc_end])
        if not inserted and chunk_type == b"IHDR":
            crc = zlib.crc32(b"grAb" + grab_data) & 0xFFFFFFFF
            result.extend(struct.pack(">I", len(grab_data)))
            result.extend(b"grAb")
            result.extend(grab_data)
            result.extend(struct.pack(">I", crc))
            inserted = True
        idx = chunk_crc_end
    if idx < len(data):
        result.extend(data[idx:])
    if not inserted:
        crc = zlib.crc32(b"grAb" + grab_data) & 0xFFFFFFFF
        result.extend(struct.pack(">I", len(grab_data)))
        result.extend(b"grAb")
        result.extend(grab_data)
        result.extend(struct.pack(">I", crc))
    return bytes(result)


def write_png_rgba(path: Path, image: DecodedGraphic) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(PNG_MAGIC)

        def write_chunk(chunk_type: bytes, chunk_data: bytes) -> None:
            handle.write(struct.pack(">I", len(chunk_data)))
            handle.write(chunk_type)
            handle.write(chunk_data)
            crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            handle.write(struct.pack(">I", crc))

        ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
        write_chunk(b"IHDR", ihdr)

        if image.has_offsets:
            grab_data = struct.pack(">ii", int(image.left_offset), int(image.top_offset))
            write_chunk(b"grAb", grab_data)

        row_stride = image.width * 4
        raw = bytearray()
        for y in range(image.height):
            raw.append(0)  # filter type 0
            start = y * row_stride
            raw.extend(image.rgba[start : start + row_stride])
        compressed = zlib.compress(bytes(raw), level=9)
        write_chunk(b"IDAT", compressed)
        write_chunk(b"IEND", b"")


def format_float(value: float) -> str:
    text = f"{value:.6f}"
    text = text.rstrip("0").rstrip(".")
    if not text:
        return "0"
    return text


def classify_marker(name: str) -> Optional[str]:
    if not name.endswith("_START") and not name.endswith("_END"):
        return None
    prefix = name.split("_")[0].upper()
    if prefix in {"F", "FF", "FL", "WF"}:
        return "flat"
    if prefix in {"S", "SP", "SS", "SPRITE", "PSPR", "J"}:
        return "sprite"
    if prefix in {"HI"}:
        return "hires"
    if prefix in {"P", "PP", "PT", "PATCH", "TX"}:
        return "patch"
    return None


def decode_patch(data: bytes, palette: Sequence[Tuple[int, int, int]]) -> Optional[DecodedGraphic]:
    if len(data) < 8:
        return None
    width, height, left_off, top_off = struct.unpack_from("<hhhh", data, 0)
    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return None
    expected_table = 8 + width * 4
    if len(data) < expected_table:
        return None
    column_offsets = struct.unpack_from("<" + "I" * width, data, 8)
    pixels = bytearray(width * height * 4)
    data_mv = memoryview(data)
    for x, offset in enumerate(column_offsets):
        if offset >= len(data):
            return None
        ptr = offset
        while True:
            if ptr >= len(data):
                return None
            top_delta = data[ptr]
            ptr += 1
            if top_delta == 255:
                break
            if ptr >= len(data):
                return None
            count = data[ptr]
            ptr += 1
            if count <= 0 or count > height + 10:  # basic sanity
                return None
            if ptr >= len(data):
                return None
            ptr += 1  # skip unused byte
            for i in range(count):
                if ptr >= len(data):
                    return None
                y = top_delta + i
                if 0 <= y < height:
                    palette_index = data[ptr]
                    r, g, b = palette[palette_index]
                    pixel_index = (y * width + x) * 4
                    pixels[pixel_index : pixel_index + 4] = bytes((r, g, b, 255))
                ptr += 1
            if ptr >= len(data):
                return None
            ptr += 1  # skip trailing zero
    return DecodedGraphic(
        width=width,
        height=height,
        rgba=bytes(pixels),
        left_offset=left_off,
        top_offset=top_off,
        has_offsets=True,
    )


def decode_flat(data: bytes, palette: Sequence[Tuple[int, int, int]]) -> Optional[DecodedGraphic]:
    length = len(data)
    if length == 0:
        return None
    size = math.isqrt(length)
    if size * size != length or size <= 0 or size > 2048:
        return None
    pixels = bytearray(size * size * 4)
    for idx, palette_index in enumerate(data):
        r, g, b = palette[palette_index]
        pixel_index = idx * 4
        pixels[pixel_index : pixel_index + 4] = bytes((r, g, b, 255))
    return DecodedGraphic(width=size, height=size, rgba=bytes(pixels))


def extract_wad_palette(wad: "WadFile") -> List[Tuple[int, int, int]]:
    for lump in wad.lumps:
        if lump.name.upper() == "PLAYPAL" and len(lump.data) >= 768:
            raw = lump.data[: 256 * 3]
            return [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, 256 * 3, 3)]
    return DEFAULT_DOOM_PALETTE.copy()


def apply_bicubic_downscale(path: Path, detail_scale: int, target_scale: int) -> None:
    """Downscale an image file using bicubic filtering while preserving format."""
    if target_scale <= 0 or detail_scale <= 0 or target_scale >= detail_scale:
        return
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        logging.warning("Pillow is required for bicubic downscaling; skipping %s", path)
        return

    ratio = target_scale / detail_scale
    try:
        with Image.open(path) as img:
            width, height = img.size
            if width == 0 or height == 0:
                return
            new_size = (
                max(1, int(round(width * ratio))),
                max(1, int(round(height * ratio))),
            )
            if new_size == img.size:
                return
            Resampling = getattr(Image, "Resampling", Image)
            downscaled = img.resize(new_size, Resampling.BICUBIC)
            img_format = img.format
            save_kwargs: Dict[str, Any] = {}
            if img_format:
                save_kwargs["format"] = img_format
        downscaled.save(path, **save_kwargs)
    except Exception as exc:  # noqa: BLE001 - best-effort downscale
        logging.warning("Failed to apply bicubic downscale for %s: %s", path, exc)


class UpscaleExecutor(Protocol):
    def upscale(self, job: TextureJob) -> Path: ...


class RealEsrganRunner:
    """Helper that wraps invocations of the realesrgan-ncnn-vulkan binary."""

    def __init__(
        self,
        executable: str,
        model_world: str,
        model_ui: str,
        tile: int = 0,
        tile_pad: int = 0,
        models_dir: Optional[Path] = None,
        gpu: str = "auto",
    ) -> None:
        self.executable = executable
        self.model_world = self._normalize_model(model_world)
        self.model_ui = self._normalize_model(model_ui)
        self.tile = max(0, tile)
        if tile_pad:
            logging.debug(
                "Ignoring requested RealESRGAN tile padding (%d); current binary does not accept -p.",
                tile_pad,
            )
        self.models_dir = models_dir.resolve() if models_dir else None
        self.gpu = gpu
        gpu_label: Optional[str] = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device_index: Optional[int] = None
                if isinstance(self.gpu, str):
                    gpu_lower = self.gpu.lower()
                    if gpu_lower in {"auto", "-1"}:
                        device_index = torch.cuda.current_device()
                    else:
                        try:
                            device_index = int(self.gpu)
                        except ValueError:
                            device_index = torch.cuda.current_device()
                else:
                    try:
                        device_index = int(self.gpu)
                    except (TypeError, ValueError):
                        device_index = torch.cuda.current_device()
                if device_index is not None and 0 <= device_index < torch.cuda.device_count():
                    gpu_label = torch.cuda.get_device_name(device_index)
        except Exception as exc:  # noqa: BLE001 - best-effort logging
            logging.debug("Unable to query CUDA device name: %s", exc)
        if gpu_label:
            logging.info("Real-ESRGAN GPU selection: %s (%s)", self.gpu, gpu_label)
        else:
            logging.info("Real-ESRGAN GPU selection: %s", self.gpu)

    @staticmethod
    def _normalize_model(model_name: str) -> str:
        normalized = model_name.strip()
        if not normalized:
            return normalized
        return Path(normalized).stem

    def model_for_category(self, category: str) -> str:
        if category in {"ui", "sprite", "character"}:
            return self.model_ui
        return self.model_world

    def run(self, input_path: Path, output_path: Path, scale: int, category: str) -> Path:
        model = self.model_for_category(category)
        cmd = [
            self.executable,
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-s",
            str(scale),
            "-n",
            model,
        ]
        if self.models_dir:
            cmd.extend(["-m", str(self.models_dir)])
        if self.tile:
            cmd.extend(["-t", str(self.tile)])
        if self.gpu and str(self.gpu).lower() != "auto":
            cmd.extend(["-g", str(self.gpu)])
        logging.debug("RealESRGAN command: %s", cmd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        try:
            completed = subprocess.run(cmd, capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"RealESRGAN executable not found: {self.executable}. "
                "Check --realesrgan-bin."
            ) from exc
        if completed.returncode != 0 or not output_path.exists():
            stderr = ""
            if completed.stderr:
                stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"RealESRGAN failed with exit code {completed.returncode}: {stderr}"
            )
        return output_path


class PillowBicubicUpscaler(UpscaleExecutor):
    """Fallback upscaler that relies on Pillow when RealESRGAN backends are unavailable."""

    def __init__(self, post_sharpen: bool = True) -> None:
        self.post_sharpen = bool(post_sharpen)
        resampling = getattr(Image, "Resampling", None)
        self._upscale_filter = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
        self._downscale_filter = getattr(resampling, "BICUBIC", Image.BICUBIC)

    def upscale(self, job: TextureJob) -> Path:
        if job.dry_run:
            return job.input_path

        detail_scale = max(1, int(job.detail_scale))
        target_scale = max(1, int(job.target_scale))
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(job.input_path) as src:
                # Copy so the file handle can close before Pillow manipulates the data.
                image = src.convert("RGBA") if src.mode == "P" else src.copy()
                original_size = (src.width, src.height)
        except Exception as exc:
            raise RuntimeError(f"Fallback upscaler failed to open {job.input_path}: {exc}") from exc

        orig_width, orig_height = original_size
        if detail_scale != 1:
            detail_size = (max(1, orig_width * detail_scale), max(1, orig_height * detail_scale))
            image = image.resize(detail_size, resample=self._upscale_filter)
        else:
            detail_size = original_size

        if target_scale != detail_scale:
            final_size = (max(1, orig_width * target_scale), max(1, orig_height * target_scale))
            image = image.resize(final_size, resample=self._downscale_filter)
        else:
            final_size = detail_size

        if self.post_sharpen:
            image = _apply_unsharp_mask_image(image, strength="default")

        try:
            image.save(job.output_path)
        except Exception as exc:
            raise RuntimeError(f"Fallback upscaler failed to save {job.output_path}: {exc}") from exc

        logging.debug(
            "Fallback Pillow upscaler wrote %s at %dx%d (detail_scale=%d, target_scale=%d)",
            job.output_path,
            final_size[0],
            final_size[1],
            detail_scale,
            target_scale,
        )
        return job.output_path


class RealEsrganUpscaler(UpscaleExecutor):
    def __init__(
        self,
        runner: Optional[RealEsrganRunner],
        lib_upscaler: Optional["LibRealESRGANUpscaler"],
        post_sharpen: bool,
        esrgan_strength: float,
        esrgan_detail_limit: float,
    ) -> None:
        if runner is None and lib_upscaler is None:
            raise ValueError("RealEsrganUpscaler requires at least one backend.")
        self.runner = runner
        self.lib_upscaler = lib_upscaler
        self.post_sharpen = bool(post_sharpen)
        self.esrgan_strength = max(0.0, float(esrgan_strength))
        self.esrgan_detail_limit = max(0.0, float(esrgan_detail_limit))
        self.sprite_edge_mode = "smooth"
        self._recent_circle_components: List[Dict[str, float]] = []
        self._recent_alpha_extension_mask = None
        self._recent_alpha_extension_orientation = None
        self._recent_triangle_components: List[Dict[str, Any]] = []
        self._recent_alpha_extension_triangles: List[Dict[str, Any]] = []
        self._recent_edge_blur_radius: float = 0.0

    @staticmethod
    def _align_model_dimension(value: int) -> int:
        base = max(value, MIN_MODEL_INPUT_DIM)
        if MIN_MODEL_INPUT_MULTIPLE > 1:
            base = int(math.ceil(base / MIN_MODEL_INPUT_MULTIPLE) * MIN_MODEL_INPUT_MULTIPLE)
        return base

    def upscale(self, job: TextureJob) -> Path:
        if job.dry_run:
            return job.input_path

        prepared_input, needs_cleanup = self._prepare_esrgan_input(job, job.input_path)
        cleanup_paths: List[Path] = []
        if needs_cleanup:
            cleanup_paths.append(prepared_input)

        prescale_result = self._ensure_target_scale(job, prepared_input)
        if (
            not isinstance(prescale_result, tuple)
            or len(prescale_result) != 2
        ):
            prescaled_input = prepared_input
            prescale_created = False
        else:
            prescaled_input, prescale_created = prescale_result

        if prescale_created:
            prepared_input = prescaled_input
            cleanup_paths.append(prepared_input)

        min_guard_result = self._ensure_minimum_model_size(prepared_input)
        if (
            isinstance(min_guard_result, tuple)
            and len(min_guard_result) == 2
            and min_guard_result[0] is not None
        ):
            min_guard_path, min_guard_created = min_guard_result
            if min_guard_created:
                prepared_input = min_guard_path
                cleanup_paths.append(prepared_input)

        keep_intermediate = bool(job.metadata.get("keep_intermediate")) if job.metadata else False
        try:
            if self.lib_upscaler is not None:
                result_path = self._run_lib_backend(job, prepared_input)
            else:
                if self.runner is None:
                    raise RuntimeError("No RealESRGAN backend is available.")
                result_path = self.runner.run(
                    prepared_input,
                    job.output_path,
                    job.detail_scale,
                    job.category,
                )
        finally:
            if not keep_intermediate:
                for temp_path in dict.fromkeys(cleanup_paths):
                    self._safe_cleanup(temp_path)

        self._finalize_esrgan_output(job, result_path)
        apply_bicubic_downscale(result_path, job.detail_scale, job.target_scale)
        self._resize_to_target(job, result_path)

        if self.post_sharpen:
            self._apply_post_sharpen_file(result_path)

        return result_path


    def _run_lib_backend(self, job: TextureJob, prepared_input: Path) -> Path:
        if self.lib_upscaler is None:
            raise RuntimeError("RealESRGAN library backend is not configured.")
        model_scale = getattr(self.lib_upscaler, "scale", job.detail_scale)
        if job.detail_scale != model_scale:
            raise ValueError(
                f"RealESRGAN lib backend expects detail scale {model_scale}, "
                f"but received {job.detail_scale}. Adjust --detail-scale accordingly."
            )
        self.lib_upscaler.upscale_image(str(prepared_input), str(job.output_path))
        try:
            from PIL import Image  # type: ignore
            from PIL import ImageChops  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Pillow is required to validate RealESRGAN output dimensions.") from exc

        with Image.open(prepared_input) as src_img:
            src_width, src_height = src_img.size
            src_rgb = src_img.convert("RGB")
        expected_width = int(round(src_width * model_scale))
        expected_height = int(round(src_height * model_scale))
        expected_size = (expected_width, expected_height)
        Resampling = getattr(Image, "Resampling", Image)
        bicubic_rgb = src_rgb.resize(expected_size, Resampling.BICUBIC)
        src_rgb.close()

        with Image.open(job.output_path) as dst_img:
            dst_width, dst_height = dst_img.size
            dst_rgb = dst_img.convert("RGB")

        if dst_width < expected_width or dst_height < expected_height:
            bicubic_rgb.close()
            dst_rgb.close()
            raise RuntimeError(
                "RealESRGAN did not enlarge the texture as expected; "
                f"output {dst_width}x{dst_height}, expected at least {expected_width}x{expected_height} "
                f"(detail_scale={model_scale})."
            )
        logging.debug(
            "RealESRGAN lib backend produced %dx%d from %dx%d using scale %s.",
            dst_width,
            dst_height,
            src_width,
            src_height,
            model_scale,
        )

        if dst_width == expected_width and dst_height == expected_height:
            if bicubic_rgb.size != dst_rgb.size:
                bicubic_rgb = bicubic_rgb.resize(dst_rgb.size, Resampling.BICUBIC)
            diff_image = ImageChops.difference(dst_rgb, bicubic_rgb)
            if diff_image.getbbox() is None:
                bicubic_rgb.close()
                dst_rgb.close()
                diff_image.close()
                raise RuntimeError(
                    "RealESRGAN output is identical to a bicubic upscale; verify that the .pth model is valid."
                )
            diff_image.close()

        dst_rgb.close()
        bicubic_rgb.close()
        return job.output_path

    def _apply_post_sharpen_file(self, path: Path) -> None:
        _apply_unsharp_mask_file(path, strength="default")

    @staticmethod
    def _apply_post_sharpen_image(image: "Image.Image") -> "Image.Image":
        return _apply_unsharp_mask_image(image, strength="default")

    def _bleed_alpha_colors(self, image: "Image.Image", alpha: "Image.Image") -> "Image.Image":
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            return image

        try:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            alpha_arr = np.asarray(alpha, dtype=np.uint8)
        except Exception:
            return image

        recent_extension = getattr(self, "_recent_alpha_extension_mask", None)
        if recent_extension is not None:
            try:
                extension_mask_arr = np.asarray(recent_extension, dtype=bool)
                if extension_mask_arr.shape != alpha_arr.shape:
                    extension_mask_arr = None
            except Exception:
                extension_mask_arr = None
        else:
            extension_mask_arr = None

        solid_threshold = 220
        fallback_threshold = 8
        known = alpha_arr >= solid_threshold
        if extension_mask_arr is not None:
            if not isinstance(known, np.ndarray):
                known = np.asarray(known, dtype=bool)
            known = known.copy()
            known[extension_mask_arr] = False
        if not known.any():
            known = alpha_arr > fallback_threshold
            if extension_mask_arr is not None:
                known = known.copy()
                known[extension_mask_arr] = False
        extension_present = bool(extension_mask_arr is not None and extension_mask_arr.any())
        if not known.any() and not extension_present:
            self._recent_alpha_extension_mask = None
            self._recent_alpha_extension_orientation = None
            return image
        if known.all() and not extension_present:
            self._recent_alpha_extension_mask = None
            self._recent_alpha_extension_orientation = None
            return image

        filled = rgb.copy()
        result_rgb = rgb.copy()
        height, width = alpha_arr.shape
        directions = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
        max_iterations = max(height, width)
        initial_known = known.copy()
        initial_known_bool = initial_known.astype(bool)
        if initial_known_bool.any():
            interior_mean_color = np.mean(rgb[initial_known_bool], axis=0)
            interior_mean_color_uint8 = np.clip(interior_mean_color + 0.5, 0, 255).astype(np.uint8)
        else:
            interior_mean_color_uint8 = np.array([0, 0, 0], dtype=np.uint8)
        for _ in range(max_iterations):
            prev_known = known.copy()
            for dy, dx in directions:
                src_y = slice(max(0, dy), height + min(0, dy))
                src_x = slice(max(0, dx), width + min(0, dx))
                dst_y = slice(max(0, -dy), height - max(0, dy))
                dst_x = slice(max(0, -dx), width - max(0, dx))
                if src_y.start >= src_y.stop or src_x.start >= src_x.stop:
                    continue
                src_mask = prev_known[src_y, src_x]
                dst_mask = known[dst_y, dst_x]
                needs = (~dst_mask) & src_mask
                if not needs.any():
                    continue
                src_rgb = filled[src_y, src_x]
                dst_rgb = filled[dst_y, dst_x]
                dst_rgb[needs] = src_rgb[needs]
                dst_mask[needs] = True
            if np.array_equal(prev_known, known):
                break

        triangle_records = getattr(self, "_recent_alpha_extension_triangles", None)
        triangle_mask: Optional["np.ndarray"] = None
        extension_mask_bool = np.zeros((height, width), dtype=bool)
        extension_replace_mask = np.zeros((height, width), dtype=bool)
        default_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        triangle_fill_mask_result: Optional["np.ndarray"] = None

        def _sample_interior_color(py: int, px: int, preferred_offsets: Iterable[Tuple[int, int]]) -> "np.ndarray":
            max_step = 6
            for step in range(1, max_step + 1):
                found_color: Optional["np.ndarray"] = None
                for dy, dx in preferred_offsets:
                    sy = py + dy * step
                    sx = px + dx * step
                    if 0 <= sy < height and 0 <= sx < width and initial_known_bool[sy, sx]:
                        found_color = rgb[sy, sx]
                        break
                if found_color is not None:
                    return found_color
            max_radius = 4
            for radius in range(1, max_radius + 1):
                y0 = max(0, py - radius)
                y1 = min(height, py + radius + 1)
                x0 = max(0, px - radius)
                x1 = min(width, px + radius + 1)
                sub_mask = initial_known_bool[y0:y1, x0:x1]
                if not sub_mask.any():
                    continue
                coords = np.argwhere(sub_mask)
                if coords.size == 0:
                    continue
                best_coord: Optional[Tuple[int, int]] = None
                best_dist = float("inf")
                for oy, ox in coords:
                    gy = y0 + int(oy)
                    gx = x0 + int(ox)
                    dy = gy - py
                    dx = gx - px
                    dist = dy * dy + dx * dx
                    if dist < best_dist:
                        best_dist = dist
                        best_coord = (gy, gx)
                if best_coord is not None:
                    gy, gx = best_coord
                    return rgb[gy, gx]
            return interior_mean_color_uint8

        triangle_mask: Optional["np.ndarray"] = None
        if extension_present:
            extension_mask_bool = np.asarray(extension_mask_arr, dtype=bool)
            extension_replace_mask = np.zeros_like(extension_mask_bool, dtype=bool)
            if triangle_records:
                triangle_fill_mask = np.zeros((height, width), dtype=bool)
                offsets_map = {
                    1: [(-1, 0), (0, -1), (-1, -1)],
                    2: [(-1, 0), (0, 1), (-1, 1)],
                    3: [(1, 0), (0, -1), (1, -1)],
                    4: [(1, 0), (0, 1), (1, 1)],
                }
                for tri in triangle_records:
                    try:
                        apex_y, apex_x = tri["apex"]
                        horizontal_y, horizontal_x = tri["horizontal"]
                        vertical_y, vertical_x = tri["vertical"]
                        orient_val = int(tri.get("orientation", 0))
                    except (KeyError, TypeError, ValueError):
                        continue
                    apex_x = float(apex_x)
                    apex_y = float(apex_y)
                    horizontal_x = float(horizontal_x)
                    horizontal_y = float(horizontal_y)
                    vertical_x = float(vertical_x)
                    vertical_y = float(vertical_y)
                    verts_x = (apex_x, horizontal_x, vertical_x)
                    verts_y = (apex_y, horizontal_y, vertical_y)
                    x_min = max(0, int(math.floor(min(verts_x))))
                    x_max = min(width, int(math.ceil(max(verts_x))))
                    y_min = max(0, int(math.floor(min(verts_y))))
                    y_max = min(height, int(math.ceil(max(verts_y))))
                    if x_min >= x_max or y_min >= y_max:
                        continue
                    denom = ((horizontal_y - vertical_y) * (apex_x - vertical_x)) + (
                        (vertical_x - horizontal_x) * (apex_y - vertical_y)
                    )
                    if abs(denom) < 1e-8:
                        continue
                    offsets_dir = offsets_map.get(orient_val, default_offsets)
                    for yy in range(y_min, y_max):
                        py = yy + 0.5
                        for xx in range(x_min, x_max):
                            px = xx + 0.5
                            u = ((horizontal_y - vertical_y) * (px - vertical_x) + (vertical_x - horizontal_x) * (py - vertical_y)) / denom
                            v = ((vertical_y - apex_y) * (px - vertical_x) + (apex_x - vertical_x) * (py - vertical_y)) / denom
                            w = 1.0 - u - v
                            if u < -1e-4 or v < -1e-4 or w < -1e-4:
                                continue
                            color_sample = _sample_interior_color(int(yy), int(xx), offsets_dir)
                            result_rgb[yy, xx] = color_sample
                            triangle_fill_mask[yy, xx] = True
                if triangle_fill_mask.any():
                    extension_replace_mask |= triangle_fill_mask
                    triangle_mask = triangle_fill_mask
                    triangle_fill_mask_result = triangle_fill_mask.copy()
                else:
                    triangle_mask = None
            else:
                triangle_mask = None

            if triangle_mask is None or not triangle_mask.any():
                if extension_mask_bool.any():
                    for yy, xx in np.argwhere(extension_mask_bool):
                        color_sample = _sample_interior_color(int(yy), int(xx), default_offsets)
                        result_rgb[int(yy), int(xx)] = color_sample
                    extension_replace_mask |= extension_mask_bool

        if triangle_fill_mask_result is not None:
            extension_mask_final = triangle_fill_mask_result

        final_known = known.copy()
        base_sample_mask = initial_known
        if not base_sample_mask.any():
            base_sample_mask = alpha_arr > fallback_threshold
        if not final_known.all():
            sample = filled[base_sample_mask] if base_sample_mask.any() else filled[alpha_arr > fallback_threshold]
            if sample.size > 0:
                avg = sample.mean(axis=0)
                filled[~final_known] = avg.astype(np.uint8)

        self._recent_edge_blur_radius = 0.0

        replace_mask = alpha_arr < solid_threshold
        if extension_present and extension_replace_mask.any():
            replace_mask = np.logical_and(replace_mask, ~extension_replace_mask)
        result_rgb[replace_mask] = filled[replace_mask]
        result = Image.fromarray(result_rgb.astype(np.uint8), "RGB")
        result.putalpha(alpha)
        self._recent_alpha_extension_mask = None
        self._recent_alpha_extension_orientation = None
        self._recent_alpha_extension_triangles = []
        return result

    def _prepare_esrgan_input(self, job: TextureJob, path: Path) -> Tuple[Path, bool]:
        try:
            from PIL import Image, ImageStat  # type: ignore
        except ImportError:
            return path, False

        try:
            with Image.open(path) as img:
                img_rgba = img.convert("RGBA")
        except Exception:
            return path, False

        alpha = img_rgba.split()[3]
        if alpha.getextrema()[0] >= 255:
            return path, False

        rgb = img_rgba.convert("RGB")
        if alpha.getbbox() is None:
            fill_color = (0, 0, 0)
        else:
            stat = ImageStat.Stat(rgb, alpha)
            fill_color = tuple(int(round(v)) for v in stat.mean)

        fill = Image.new("RGB", img_rgba.size, fill_color)
        composite_rgb = Image.composite(rgb, fill, alpha)
        composite_rgba = Image.merge("RGBA", (*composite_rgb.split(), alpha))
        composite_rgba = self._bleed_alpha_colors(composite_rgba, alpha)

        fd, temp_name = tempfile.mkstemp(
            prefix="esr_prep_",
            suffix=path.suffix,
            dir=str(path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        composite_rgba.save(temp_path)
        return temp_path, True

    def _ensure_target_scale(self, job: TextureJob, path: Path) -> Tuple[Path, bool]:
        if job.detail_scale <= 0:
            return path, False

        try:
            from PIL import Image  # type: ignore
        except ImportError:
            logging.warning("Pillow is required to pre-scale textures; skipping %s", path)
            return path, False

        scale_ratio = 1.0
        if job.target_scale > 0:
            scale_ratio = max(scale_ratio, job.target_scale / job.detail_scale)

        try:
            with Image.open(path) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    return path, False

                min_ratio_width = 1.0
                min_ratio_height = 1.0
                if width < MIN_MODEL_INPUT_DIM:
                    min_ratio_width = MIN_MODEL_INPUT_DIM / max(width, 1)
                if height < MIN_MODEL_INPUT_DIM:
                    min_ratio_height = MIN_MODEL_INPUT_DIM / max(height, 1)

                scale_ratio = max(scale_ratio, min_ratio_width, min_ratio_height)

                if math.isclose(scale_ratio, 1.0, rel_tol=1e-6):
                    return path, False

                new_size = (
                    self._align_model_dimension(int(round(width * scale_ratio))),
                    self._align_model_dimension(int(round(height * scale_ratio))),
                )
                if new_size == img.size:
                    return path, False
                Resampling = getattr(Image, "Resampling", Image)
                resized = img.resize(new_size, Resampling.BICUBIC)
                img_format = img.format
        except Exception as exc:
            logging.warning("Failed to pre-scale %s by %.3fx: %s", path, scale_ratio, exc)
            return path, False

        save_kwargs: Dict[str, Any] = {}
        if img_format:
            save_kwargs["format"] = img_format
        fd, temp_name = tempfile.mkstemp(
            prefix="esr_prescale_",
            suffix=path.suffix,
            dir=str(path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            resized.save(temp_path, **save_kwargs)
        finally:
            try:
                resized.close()
            except Exception:
                pass
        logging.debug(
            "Pre-scaled %s to %dx%d (ratio %.3f) before AI inference (was %dx%d).",
            path,
            new_size[0],
            new_size[1],
            scale_ratio,
            width,
            height,
        )
        return temp_path, True

    def _ensure_minimum_model_size(self, path: Path) -> Tuple[Optional[Path], bool]:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return path, False

        try:
            with Image.open(path) as img:
                width, height = img.size
                if width >= MIN_MODEL_INPUT_DIM and height >= MIN_MODEL_INPUT_DIM:
                    return path, False
                new_width = self._align_model_dimension(width)
                new_height = self._align_model_dimension(height)
                if new_width == width and new_height == height:
                    return path, False
                Resampling = getattr(Image, "Resampling", Image)
                resize_filter = getattr(Resampling, "BICUBIC", Image.BICUBIC)
                resized = img.resize((new_width, new_height), resize_filter)
                img_format = img.format
        except Exception as exc:
            logging.warning("Failed to enforce minimum model size for %s: %s", path, exc)
            return path, False

        save_kwargs: Dict[str, Any] = {}
        if img_format:
            save_kwargs["format"] = img_format
        fd, temp_name = tempfile.mkstemp(
            prefix="esr_guard_",
            suffix=path.suffix,
            dir=str(path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            resized.save(temp_path, **save_kwargs)
        finally:
            try:
                resized.close()
            except Exception:
                pass
        logging.debug(
            "Expanded %s to %dx%d to satisfy minimum model requirements (was %dx%d).",
            path,
            new_width,
            new_height,
            width,
            height,
        )
        return temp_path, True

    def _resize_to_target(self, job: TextureJob, path: Path) -> None:
        metadata = job.metadata or {}
        try:
            original_width = int(metadata.get("original_width", 0) or 0)
            original_height = int(metadata.get("original_height", 0) or 0)
        except Exception:
            original_width = 0
            original_height = 0
        if original_width <= 0 or original_height <= 0:
            return

        expected_width = max(1, int(round(original_width * job.target_scale)))
        expected_height = max(1, int(round(original_height * job.target_scale)))

        try:
            from PIL import Image  # type: ignore
        except ImportError:
            logging.warning("Pillow is required to adjust final texture size; skipping %s", path)
            return

        try:
            with Image.open(path) as img:
                width, height = img.size
                if width == expected_width and height == expected_height:
                    return
                Resampling = getattr(Image, "Resampling", Image)
                resized = img.resize((expected_width, expected_height), Resampling.LANCZOS)
                img_format = img.format
            save_kwargs: Dict[str, Any] = {}
            if img_format:
                save_kwargs["format"] = img_format
            resized.save(path, **save_kwargs)
            try:
                resized.close()
            except Exception:
                pass
            logging.debug(
                "Resized %s to match target dimensions %dx%d (was %dx%d).",
                path,
                expected_width,
                expected_height,
                width,
                height,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to adjust final size for %s: %s", path, exc)

    def _apply_esr_smoothing(
        self,
        source_path: Path,
        esr_rgba: "Image.Image",
        alpha: "Image.Image",
    ) -> Tuple["Image.Image", "Image.Image"]:
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return esr_rgba, alpha
        try:
            with Image.open(source_path) as orig:
                source_rgb = orig.convert("RGB")
        except Exception:
            return esr_rgba, alpha
        Resampling = getattr(Image, "Resampling", Image)
        source_rgb = source_rgb.resize(esr_rgba.size, Resampling.BICUBIC)
        esr_rgb = esr_rgba.convert("RGB")
        source_np = np.asarray(source_rgb, dtype=np.float32)
        esr_np = np.asarray(esr_rgb, dtype=np.float32)
        diff = esr_np - source_np
        if self.esrgan_detail_limit > 0.0:
            limit = float(self.esrgan_detail_limit)
            diff = np.clip(diff, -limit, limit)
        blended_np = source_np + diff * float(self.esrgan_strength)
        blended_np = np.clip(blended_np, 0.0, 255.0).astype(np.uint8)
        blended_rgb = Image.fromarray(blended_np, "RGB")
        blended_rgba = blended_rgb.convert("RGBA")
        alpha_copy = alpha.copy()
        blended_rgba.putalpha(alpha_copy)
        blended_rgba = self._bleed_alpha_colors(blended_rgba, alpha_copy)
        return blended_rgba, alpha_copy

    def _finalize_esrgan_output(self, job: TextureJob, result_path: Path) -> None:
        sprite_categories = {"sprite", "character", "ui"}
        alpha_mode = self.sprite_edge_mode if job.category in sprite_categories else "smooth"
        target_delta_scale = max(1, job.target_scale)
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            self._restore_original_alpha(
                job.input_path,
                result_path,
                edge_mode=alpha_mode,
                delta_scale=target_delta_scale,
            )
            return
        try:
            with Image.open(result_path) as img:
                esr_rgba = img.convert("RGBA")
        except Exception:
            self._restore_original_alpha(
                job.input_path,
                result_path,
                edge_mode=alpha_mode,
                delta_scale=target_delta_scale,
            )
            return
        delta_path: Optional[Path] = None
        if job.metadata:
            delta_hint = job.metadata.get("alpha_delta_path")
            if delta_hint:
                delta_path = Path(delta_hint)
        original_alpha = self._load_scaled_alpha(
            job.input_path,
            esr_rgba.size,
            delta_path,
            delta_scale=target_delta_scale,
            edge_mode=alpha_mode,
        )
        if original_alpha is not None:
            esr_rgba.putalpha(original_alpha)
            alpha = original_alpha
        else:
            alpha = esr_rgba.split()[3]
        esr_rgba, alpha = self._apply_esr_smoothing(job.input_path, esr_rgba, alpha)
        esr_rgba.putalpha(alpha)
        esr_rgba = self._bleed_alpha_colors(esr_rgba, alpha)
        esr_rgba.save(result_path)
        self._restore_original_alpha(
            job.input_path,
            result_path,
            edge_mode=alpha_mode,
            delta_scale=target_delta_scale,
        )

    def _refine_alpha_with_marching_squares(
        self,
        mask: "Image.Image",
    ) -> Tuple[Optional["Image.Image"], Optional["np.ndarray"]]:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            logging.debug("Alpha corner smoothing skipped: numpy not available.")
            return None, None

        mask_raw = np.asarray(mask, dtype=np.uint8)
        if mask_raw.size == 0:
            logging.debug("Alpha corner smoothing skipped: mask is empty.")
            return None, None

        solid_threshold = int(0.9 * 255)
        transparent_threshold = int(0.12 * 255)

        solid = mask_raw >= solid_threshold
        if not solid.any():
            logging.debug("Alpha corner smoothing skipped: mask has no solid pixels.")
            return None, None
        transparent = mask_raw <= transparent_threshold
        if not transparent.any():
            logging.debug("Alpha corner smoothing skipped: mask lacks transparent pixels.")
            return None, None

        solid_count = int(np.count_nonzero(solid))
        transparent_count = int(np.count_nonzero(transparent))
        extreme_pixels = solid_count + transparent_count
        extreme_ratio = extreme_pixels / float(mask_raw.size)
        if extreme_ratio < 0.35:
            logging.debug(
                "Alpha corner smoothing skipped: insufficient strongly-defined alpha regions (%.3f extremes ratio).",
                extreme_ratio,
            )
            return None, None

        solid_ratio = solid_count / float(mask_raw.size)
        transparent_ratio = transparent_count / float(mask_raw.size)
        if solid_ratio < 0.2 and transparent_ratio > 0.4:
            logging.debug(
                "Alpha corner smoothing skipped: alpha dominated by translucency (solid_ratio=%.3f, transparent_ratio=%.3f).",
                solid_ratio,
                transparent_ratio,
            )
            return None, None

        strong_edges = self._measure_alpha_edge_strength(mask_raw)
        total_edges = (
            max(0, (mask_raw.shape[0] - 1) * mask_raw.shape[1])
            + max(0, mask_raw.shape[0] * (mask_raw.shape[1] - 1))
        )
        total_edges = max(1, total_edges)
        strong_ratio = strong_edges / float(total_edges)
        perimeter_estimate = max(1, 2 * (mask_raw.shape[0] + mask_raw.shape[1]))
        perimeter_ratio = strong_edges / float(perimeter_estimate)
        if strong_edges < 4 or (strong_ratio < 0.002 and perimeter_ratio < 0.5):
            logging.debug(
                "Alpha corner smoothing skipped: insufficient sharp alpha transitions (%d edges, %.5f ratio, perimeter=%.3f).",
                strong_edges,
                strong_ratio,
                perimeter_ratio,
            )
            return None, None

        # Work on a copy so we can detect which pixels change.
        refined = solid.copy()
        orientation = np.zeros_like(refined, dtype=np.uint8)

        # Helpful slices for 2x2 block tests.
        tl = solid[:-1, :-1]
        tr = solid[:-1, 1:]
        bl = solid[1:, :-1]
        br = solid[1:, 1:]

        # Missing bottom-right corner: fill diagonal pixel.
        mask_br = tl & tr & bl & (~br)
        refined[1:, 1:] |= mask_br
        if mask_br.any():
            orientation_slice = orientation[1:, 1:]
            orientation_slice[mask_br] = 1

        # Missing bottom-left corner.
        mask_bl = tl & tr & (~bl) & br
        refined[1:, :-1] |= mask_bl
        if mask_bl.any():
            orientation_slice = orientation[1:, :-1]
            orientation_slice[mask_bl] = 2

        # Missing top-right corner.
        mask_tr = tl & (~tr) & bl & br
        refined[:-1, 1:] |= mask_tr
        if mask_tr.any():
            orientation_slice = orientation[:-1, 1:]
            orientation_slice[mask_tr] = 3

        # Missing top-left corner.
        mask_tl = (~tl) & tr & bl & br
        refined[:-1, :-1] |= mask_tl
        if mask_tl.any():
            orientation_slice = orientation[:-1, :-1]
            orientation_slice[mask_tl] = 4

        changed = np.count_nonzero(refined ^ solid)
        logging.debug("Alpha corner smoothing added %d pixel(s).", changed)
        if changed == 0:
            return None, None

        result = np.where(refined, 255, 0).astype(np.uint8)
        return Image.fromarray(result, "L"), orientation

    def _measure_alpha_edge_strength(self, mask_raw: "np.ndarray") -> int:
        import numpy as np  # type: ignore

        if mask_raw.ndim != 2:
            mask_arr = np.asarray(mask_raw, dtype=np.uint8)
            if mask_arr.ndim != 2:
                mask_arr = mask_arr.reshape(mask_arr.shape[0], -1)
        else:
            mask_arr = mask_raw

        mask_int = mask_arr.astype(np.int16, copy=False)
        horizontal = np.abs(np.diff(mask_int, axis=1))
        vertical = np.abs(np.diff(mask_int, axis=0))
        strong_threshold = 80
        horizontal_count = int(np.count_nonzero(horizontal >= strong_threshold))
        vertical_count = int(np.count_nonzero(vertical >= strong_threshold))
        return horizontal_count + vertical_count

    def _render_alpha_delta_map(
        self,
        mask: "np.ndarray",
        target_shape: Optional[Tuple[int, int]],
        scale_factor: int,
        orientation_map: Optional["np.ndarray"] = None,
        base_mask: Optional["np.ndarray"] = None,
        *,
        enable_circle_fill: bool = True,
    ) -> "Image.Image":
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim != 2:
            mask_bool = mask_bool.astype(bool)
            if mask_bool.size == 0:
                mask_bool = np.zeros((1, 1), dtype=bool)
            mask_bool = mask_bool.reshape(mask_bool.shape[0], -1)

        base_h, base_w = mask_bool.shape
        if base_h <= 0 or base_w <= 0:
            base_h = base_w = 1

        if orientation_map is None:
            orientation = np.zeros((base_h, base_w), dtype=np.uint8)
        else:
            orientation = np.asarray(orientation_map, dtype=np.uint8)
            if orientation.shape != mask_bool.shape:
                orientation_resized = np.zeros((base_h, base_w), dtype=np.uint8)
                copy_h = min(orientation.shape[0], base_h)
                copy_w = min(orientation.shape[1], base_w)
                orientation_resized[:copy_h, :copy_w] = orientation[:copy_h, :copy_w]
                orientation = orientation_resized

        if base_mask is not None:
            base_arr = np.asarray(base_mask, dtype=np.uint8)
            if base_arr.shape != mask_bool.shape:
                base_bool = np.zeros((base_h, base_w), dtype=bool)
                copy_h = min(base_arr.shape[0], base_h)
                copy_w = min(base_arr.shape[1], base_w)
                base_bool[:copy_h, :copy_w] = base_arr[:copy_h, :copy_w] >= 128
            else:
                base_bool = base_arr >= 128
        else:
            base_bool = mask_bool

        scale_factor = max(1, int(round(scale_factor)))
        scale_threshold = float(scale_factor)

        aa_scale = max(2, min(16, scale_factor * 4))
        desired_h = max(1, (target_shape[0] if target_shape else base_h) * scale_factor)
        desired_w = max(1, (target_shape[1] if target_shape else base_w) * scale_factor)
        aa_height = max(1, desired_h * aa_scale)
        aa_width = max(1, desired_w * aa_scale)

        overlay_buffer = np.zeros((aa_height, aa_width), dtype=np.float32)
        cell_h = aa_height / float(base_h)
        cell_w = aa_width / float(base_w)

        # Identify small isolated blobs (including 2x2 squares) with no orientation metadata
        # and replace them with a rounded, anti-aliased circle sized to the target scale.
        circle_components: List[Dict[str, float]] = []
        skip_orientation = np.zeros((base_h, base_w), dtype=bool)
        if enable_circle_fill:
            analysis_bool = base_bool
            if analysis_bool.shape != mask_bool.shape:
                analysis_bool = mask_bool
            visited = np.zeros((base_h, base_w), dtype=bool)
            size_limit = max(1, scale_factor)
            square_dim = max(1, int(round(scale_factor / 2.0)))
            max_dim = max(1, square_dim)
            max_pixels_cluster = max(1, square_dim * square_dim)
            for y in range(base_h):
                for x in range(base_w):
                    if not analysis_bool[y, x] or visited[y, x]:
                        continue
                    stack = [(y, x)]
                    visited[y, x] = True
                    component: List[Tuple[int, int]] = []
                    min_y = max_y = y
                    min_x = max_x = x
                    while stack:
                        cy, cx = stack.pop()
                        component.append((cy, cx))
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if (
                                0 <= ny < base_h
                                and 0 <= nx < base_w
                                and analysis_bool[ny, nx]
                                and not visited[ny, nx]
                            ):
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                    comp_size = len(component)
                    if comp_size == 0 or comp_size > max_pixels_cluster:
                        continue
                    bbox_h = max_y - min_y + 1
                    bbox_w = max_x - min_x + 1
                    if bbox_h != bbox_w or bbox_h > max_dim or bbox_w > max_dim:
                        continue
                    if comp_size != bbox_h * bbox_w:
                        continue
                    component_set = {tuple(pixel) for pixel in component}
                    has_orthogonal_neighbor = False
                    for cy, cx in component:
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if (
                                0 <= ny < base_h
                                and 0 <= nx < base_w
                                and analysis_bool[ny, nx]
                                and (ny, nx) not in component_set
                            ):
                                has_orthogonal_neighbor = True
                                break
                        if has_orthogonal_neighbor:
                            break
                    if has_orthogonal_neighbor:
                        continue

                    center_y = (min_y + max_y + 1) * 0.5 * cell_h
                    center_x = (min_x + max_x + 1) * 0.5 * cell_w
                    radius_pixels = max(0.5, float(scale_factor) * 0.5)
                    radius_aa = radius_pixels * aa_scale
                    radius_sq = radius_aa * radius_aa
                    y_min_aa = max(0, int(math.floor(center_y - radius_aa - 1)))
                    y_max_aa = min(aa_height, int(math.ceil(center_y + radius_aa + 1)))
                    x_min_aa = max(0, int(math.floor(center_x - radius_aa - 1)))
                    x_max_aa = min(aa_width, int(math.ceil(center_x + radius_aa + 1)))

                    circle_components.append(
                        {
                            "center_y_norm": center_y / float(aa_height),
                            "center_x_norm": center_x / float(aa_width),
                            "radius_y_norm": radius_aa / float(aa_height),
                            "radius_x_norm": radius_aa / float(aa_width),
                            "center_y_aa": center_y,
                            "center_x_aa": center_x,
                            "radius_sq": radius_sq,
                            "radius_aa": radius_aa,
                            "y_min": y_min_aa,
                            "y_max": y_max_aa,
                            "x_min": x_min_aa,
                            "x_max": x_max_aa,
                        }
                    )
                    for cy, cx in component:
                        if mask_bool[cy, cx]:
                            skip_orientation[cy, cx] = True
                    for sy in range(y_min_aa, y_max_aa):
                        dy = (sy + 0.5) - center_y
                        dy_sq = dy * dy
                        if dy_sq >= radius_sq:
                            continue
                        for sx in range(x_min_aa, x_max_aa):
                            dx = (sx + 0.5) - center_x
                            dist_sq = dx * dx + dy_sq
                            if dist_sq >= radius_sq:
                                continue
                            if overlay_buffer[sy, sx] < 255.0:
                                overlay_buffer[sy, sx] = 255.0

        triangle_components: List[Dict[str, float]] = []
        orientation_config = {
            1: {
                "apex_offset": (0.0, 0.0),
                "h_step": (0, 1),
                "h_inside": [(-1, 0)],
                "h_dir": 1,
                "v_step": (1, 0),
                "v_inside": [(0, -1)],
                "v_dir": 1,
            },
            2: {
                "apex_offset": (1.0, 0.0),
                "h_step": (0, -1),
                "h_inside": [(-1, 0)],
                "h_dir": -1,
                "v_step": (1, 0),
                "v_inside": [(0, 1)],
                "v_dir": 1,
            },
            3: {
                "apex_offset": (0.0, 1.0),
                "h_step": (0, 1),
                "h_inside": [(1, 0)],
                "h_dir": 1,
                "v_step": (-1, 0),
                "v_inside": [(0, -1)],
                "v_dir": -1,
            },
            4: {
                "apex_offset": (1.0, 1.0),
                "h_step": (0, -1),
                "h_inside": [(1, 0)],
                "h_dir": -1,
                "v_step": (-1, 0),
                "v_inside": [(0, 1)],
                "v_dir": -1,
            },
        }

        def _edge_run(
            start_y: int,
            start_x: int,
            step_y: int,
            step_x: int,
            inside_offsets: Iterable[Tuple[int, int]],
        ) -> int:
            length = 0
            py = start_y
            px = start_x
            while True:
                py += step_y
                px += step_x
                if not (0 <= py < base_h and 0 <= px < base_w):
                    break
                if base_bool[py, px]:
                    break
                inside_valid = False
                for dy, dx in inside_offsets:
                    iy = py + dy
                    ix = px + dx
                    if 0 <= iy < base_h and 0 <= ix < base_w and base_bool[iy, ix]:
                        inside_valid = True
                        break
                if not inside_valid:
                    break
                length += 1
            return length

        coords = np.argwhere(mask_bool)
        if coords.size == 0 and not circle_components:
            self._recent_circle_components = circle_components
            return Image.new("L", (desired_w, desired_h), 0)

        for y, x in coords:
            if skip_orientation[y, x]:
                continue
            orient = int(orientation[y, x])
            params = orientation_config.get(orient)
            if not params:
                continue

            horizontal_len = _edge_run(
                y,
                x,
                params["h_step"][0],
                params["h_step"][1],
                params["h_inside"],
            )
            vertical_len = _edge_run(
                y,
                x,
                params["v_step"][0],
                params["v_step"][1],
                params["v_inside"],
            )

            horizontal_units = (1 + horizontal_len) / 2.0
            vertical_units = (1 + vertical_len) / 2.0

            extend_horizontal = horizontal_units > scale_threshold and vertical_units <= scale_threshold
            extend_vertical = vertical_units > scale_threshold and horizontal_units <= scale_threshold
            if horizontal_units > scale_threshold and vertical_units > scale_threshold:
                extend_horizontal = extend_vertical = False

            base_horizontal = min(float(scale_factor), 1 + horizontal_len)
            base_vertical = min(float(scale_factor), 1 + vertical_len)

            horizontal_extent = 1 + horizontal_len if extend_horizontal else base_horizontal
            vertical_extent = 1 + vertical_len if extend_vertical else base_vertical

            max_horizontal = 0.1 * float(base_w)
            max_vertical = 0.1 * float(base_h)
            horizontal_extent = min(horizontal_extent, max_horizontal)
            vertical_extent = min(vertical_extent, max_vertical)

            if horizontal_extent <= 0.0 or vertical_extent <= 0.0:
                continue

            apex_x = x + params["apex_offset"][0]
            apex_y = y + params["apex_offset"][1]

            horiz_sign = params["h_dir"]
            vert_sign = params["v_dir"]

            horizontal_point_x = float(np.clip(apex_x + horiz_sign * horizontal_extent, 0.0, base_w))
            horizontal_point_y = float(np.clip(apex_y, 0.0, base_h))
            vertical_point_x = float(np.clip(apex_x, 0.0, base_w))
            vertical_point_y = float(np.clip(apex_y + vert_sign * vertical_extent, 0.0, base_h))

            base_w_safe = float(base_w) if base_w > 0 else 1.0
            base_h_safe = float(base_h) if base_h > 0 else 1.0
            triangle_components.append(
                {
                    "orientation": float(orient),
                    "apex_x_norm": float(apex_x) / base_w_safe,
                    "apex_y_norm": float(apex_y) / base_h_safe,
                    "horizontal_x_norm": float(horizontal_point_x) / base_w_safe,
                    "horizontal_y_norm": float(horizontal_point_y) / base_h_safe,
                    "vertical_x_norm": float(vertical_point_x) / base_w_safe,
                    "vertical_y_norm": float(vertical_point_y) / base_h_safe,
                }
            )

            apex_x_aa = apex_x * cell_w
            apex_y_aa = apex_y * cell_h
            horiz_x_aa = horizontal_point_x * cell_w
            vert_y_aa = vertical_point_y * cell_h

            x_min = int(np.floor(min(apex_x_aa, horiz_x_aa, vertical_point_x * cell_w)))
            x_max = int(np.ceil(max(apex_x_aa, horiz_x_aa, vertical_point_x * cell_w)))
            y_min = int(np.floor(min(apex_y_aa, horizontal_point_y * cell_h, vert_y_aa)))
            y_max = int(np.ceil(max(apex_y_aa, horizontal_point_y * cell_h, vert_y_aa)))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(aa_width, x_max)
            y_max = min(aa_height, y_max)

            span_x = horiz_x_aa - apex_x_aa
            span_y = vert_y_aa - apex_y_aa
            if span_x == 0.0 or span_y == 0.0:
                continue

            span_x_abs = abs(span_x)
            span_y_abs = abs(span_y)

            for sy in range(y_min, y_max):
                cy = (sy + 0.5 - apex_y_aa) / span_y_abs
                if vert_sign < 0:
                    cy = -cy
                if cy < 0.0 or cy > 1.0:
                    continue
                for sx in range(x_min, x_max):
                    cx = (sx + 0.5 - apex_x_aa) / span_x_abs
                    if horiz_sign < 0:
                        cx = -cx
                    if cx < 0.0 or cx > 1.0 or cx + cy > 1.0:
                        continue
                    value = max(0.0, 1.0 - (cx + cy))
                    # bias toward higher opacity so the apex is more solid
                    value = min(1.0, (value * 4))
                    if value <= 0.0:
                        continue
                    current = overlay_buffer[sy, sx]
                    candidate = value * 255.0
                    if candidate > current:
                        overlay_buffer[sy, sx] = candidate

        overlay = Image.fromarray(np.clip(overlay_buffer, 0.0, 255.0).astype(np.uint8), "L")
        Resampling = getattr(Image, "Resampling", Image)
        if overlay.size != (desired_w, desired_h):
            overlay = overlay.resize((desired_w, desired_h), resample=Resampling.LANCZOS)

        if enable_circle_fill:
            overlay_arr = np.array(overlay, dtype=np.uint8, copy=True)

            def _dilate4(arr: "np.ndarray") -> "np.ndarray":
                result = arr.copy()
                height, width = arr.shape
                offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
                for dy, dx in offsets:
                    shifted = np.zeros_like(arr)
                    src_y = slice(max(0, -dy), height - max(0, dy))
                    dst_y = slice(max(0, dy), height - max(0, -dy))
                    src_x = slice(max(0, -dx), width - max(0, dx))
                    dst_x = slice(max(0, dx), width - max(0, -dx))
                    shifted[dst_y, dst_x] = arr[src_y, src_x]
                    result = np.maximum(result, shifted)
                return result

            overlay_dilated = _dilate4(overlay_arr)

            base_mask_img = Image.fromarray((base_bool.astype(np.uint8) * 255), "L")
            if base_mask_img.size != (desired_w, desired_h):
                base_mask_img = base_mask_img.resize((desired_w, desired_h), resample=Resampling.NEAREST)
            mask_arr = np.array(base_mask_img, dtype=np.uint8, copy=True)
            mask_dilated = _dilate4(mask_arr)

            seam_fill = np.where(mask_dilated > 0, overlay_dilated, overlay_arr)
            overlay_arr = np.maximum(overlay_arr, seam_fill)
            if circle_components:
                height, width = overlay_arr.shape
                circle_overlay = np.zeros_like(overlay_arr, dtype=np.uint8)
                for comp in circle_components:
                    try:
                        center_y_norm = float(comp["center_y_norm"])
                        center_x_norm = float(comp["center_x_norm"])
                        radius_y_norm = float(comp["radius_y_norm"])
                        radius_x_norm = float(comp["radius_x_norm"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    center_y = center_y_norm * height
                    center_x = center_x_norm * width
                    radius_y = max(1e-6, radius_y_norm * height)
                    radius_x = max(1e-6, radius_x_norm * width)
                    radius_sq = max(radius_y, radius_x) ** 2
                    y_min = max(0, int(math.floor(center_y - radius_y - 1)))
                    y_max = min(height, int(math.ceil(center_y + radius_y + 1)))
                    x_min = max(0, int(math.floor(center_x - radius_x - 1)))
                    x_max = min(width, int(math.ceil(center_x + radius_x + 1)))
                    if y_min >= y_max or x_min >= x_max:
                        continue
                    region = circle_overlay[y_min:y_max, x_min:x_max]
                    for sy in range(region.shape[0]):
                        py = y_min + sy
                        dy = (py + 0.5) - center_y
                        dy_sq = dy * dy
                        if dy_sq >= radius_sq:
                            continue
                        for sx in range(region.shape[1]):
                            px = x_min + sx
                            dx = (px + 0.5) - center_x
                            dist_sq = dx * dx + dy_sq
                            if dist_sq >= radius_sq:
                                continue
                            if region[sy, sx] < 255:
                                region[sy, sx] = 255
                overlay_arr = np.maximum(overlay_arr, circle_overlay)
                circle_overlay_img = Image.fromarray(circle_overlay, "L")
            else:
                circle_overlay_img = None
            overlay = Image.fromarray(overlay_arr, "L")
            if circle_overlay_img is not None:
                overlay = ImageChops.lighter(overlay, circle_overlay_img)
                overlay_arr = np.array(overlay, dtype=np.uint8, copy=True)
        else:
            circle_components = []
        self._recent_circle_components = circle_components
        self._recent_triangle_components = triangle_components
        return overlay

    def _write_alpha_delta(
        self,
        path: Path,
        changed_mask: "np.ndarray",
        target_shape: Optional[Tuple[int, int]] = None,
        scale_factor: int = 1,
        orientation_map: Optional["np.ndarray"] = None,
        base_mask: Optional["np.ndarray"] = None,
    ) -> Optional["Image.Image"]:
        try:
            import numpy as np  # type: ignore
            path = Path(path)
            delta_img = self._render_alpha_delta_map(
                changed_mask,
                target_shape,
                scale_factor,
                orientation_map=orientation_map,
                base_mask=base_mask,
                enable_circle_fill=False,
            )
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            delta_img.save(path)
            logging.debug(
                "Alpha delta saved to %s (%d changed pixel(s)).",
                path,
                int(np.asarray(changed_mask, dtype=bool).sum()),
            )
            return delta_img
        except Exception as exc:
            logging.debug("Alpha delta write failed for %s: %s", path, exc)
            return None

    def _load_scaled_alpha(
        self,
        original_path: Path,
        size: Tuple[int, int],
        delta_path: Optional[Path] = None,
        delta_scale: int = 1,
        edge_mode: str = "smooth",
    ) -> Optional["Image.Image"]:
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return None

        try:
            with Image.open(original_path) as orig:
                orig_rgba = orig.convert("RGBA")
        except Exception:
            return None

        self._recent_alpha_extension_mask = None
        self._recent_alpha_extension_orientation = None
        self._recent_triangle_components = []
        self._recent_alpha_extension_triangles = []
        self._recent_edge_blur_radius = 0.0

        delta_scale_int = max(1, int(round(delta_scale)))
        edge_mode_normalized = edge_mode.lower()
        if edge_mode_normalized not in {"smooth", "upscale"}:
            edge_mode_normalized = "smooth"

        Resampling = getattr(Image, "Resampling", Image)
        changed_mask_bool: Optional[np.ndarray] = None
        extension_mask_final: Optional[np.ndarray] = None
        triangle_fill_mask_result: Optional[np.ndarray] = None
        orientation_final: Optional[np.ndarray] = None

        def _safe_gaussian_blur(image: "Image.Image", radius: float) -> "Image.Image":
            radius = float(radius)
            if not math.isfinite(radius) or radius <= 0.0:
                return image
            attempt = radius
            while attempt > 0.0:
                try:
                    return image.filter(ImageFilter.GaussianBlur(radius=attempt))
                except ValueError as exc:
                    if "bad filter size" not in str(exc).lower():
                        raise
                    next_attempt = attempt - 0.5
                    if next_attempt <= 0.0:
                        logging.debug(
                            "Gaussian blur skipped: radius %.4f yielded an invalid filter size.", radius
                        )
                        break
                    logging.debug(
                        "Gaussian blur radius %.4f produced an invalid filter size; retrying with %.4f.",
                        attempt,
                        next_attempt,
                    )
                    attempt = next_attempt
            logging.debug(
                "Gaussian blur skipped after invalid filter sizes starting from radius %.4f.", radius
            )
            return image

        def _safe_unsharp_mask(
            image: "Image.Image",
            radius: float,
            *,
            percent: int,
            threshold: int,
        ) -> "Image.Image":
            radius = float(radius)
            if not math.isfinite(radius) or radius <= 0.0:
                return image
            attempt = radius
            while attempt > 0.0:
                try:
                    return image.filter(
                        ImageFilter.UnsharpMask(radius=attempt, percent=percent, threshold=threshold)
                    )
                except ValueError as exc:
                    if "bad filter size" not in str(exc).lower():
                        raise
                    next_attempt = attempt - 0.5
                    if next_attempt <= 0.0:
                        logging.debug(
                            "Unsharp mask skipped: radius %.4f yielded an invalid filter size.", radius
                        )
                        break
                    logging.debug(
                        "Unsharp mask radius %.4f produced an invalid filter size; retrying with %.4f.",
                        attempt,
                        next_attempt,
                    )
                    attempt = next_attempt
            logging.debug(
                "Unsharp mask skipped after invalid filter sizes starting from radius %.4f.", radius
            )
            return image

        alpha = orig_rgba.split()[3]
        if alpha.getextrema()[0] >= 255:
            if delta_path is not None:
                empty_mask = np.zeros((alpha.size[1], alpha.size[0]), dtype=bool)
                self._write_alpha_delta(
                    delta_path,
                    empty_mask,
                    (alpha.size[1], alpha.size[0]),
                    scale_factor=delta_scale_int,
                    orientation_map=None,
                )
            return None

        src_w, src_h = alpha.size
        dst_w, dst_h = size
        if dst_w <= 0 or dst_h <= 0:
            return None

        orig_shape = (src_h, src_w)

        scale_x = dst_w / max(1, src_w)
        scale_y = dst_h / max(1, src_h)
        delta_scale_int = max(1, delta_scale_int)

        delta_overlay_img: Optional[Image.Image] = None

        # Always promote to 2x before corner smoothing.
        double_w = max(1, src_w * 2)
        double_h = max(1, src_h * 2)
        alpha_double = alpha.resize((double_w, double_h), resample=Image.NEAREST)
        alpha_double_arr = np.asarray(alpha_double, dtype=np.uint8)

        refined_double, corner_map = self._refine_alpha_with_marching_squares(alpha_double)
        if corner_map is None:
            corner_map = np.zeros_like(alpha_double_arr, dtype=np.uint8)
        if refined_double is not None:
            refined_double_arr = np.asarray(refined_double, dtype=np.uint8)
            changed_mask = refined_double_arr != alpha_double_arr
            changed_mask_bool = changed_mask
            if delta_path is not None:
                delta_overlay_img = self._write_alpha_delta(
                    delta_path,
                    changed_mask,
                    orig_shape,
                    scale_factor=delta_scale_int,
                    orientation_map=corner_map,
                    base_mask=refined_double_arr,
                )
            else:
                delta_overlay_img = self._render_alpha_delta_map(
                    changed_mask,
                    orig_shape,
                    delta_scale_int,
                    orientation_map=corner_map,
                    base_mask=refined_double_arr,
                    enable_circle_fill=False,
                )
            if delta_overlay_img is None:
                delta_overlay_img = self._render_alpha_delta_map(
                    changed_mask,
                    orig_shape,
                    delta_scale_int,
                    orientation_map=corner_map,
                    base_mask=refined_double_arr,
                    enable_circle_fill=False,
                )
            working_alpha = refined_double
        else:
            null_mask = np.zeros((double_h, double_w), dtype=bool)
            if delta_path is not None:
                delta_overlay_img = self._write_alpha_delta(
                    delta_path,
                    null_mask,
                    orig_shape,
                    scale_factor=delta_scale_int,
                    orientation_map=corner_map,
                    base_mask=alpha_double_arr,
                )
            else:
                delta_overlay_img = self._render_alpha_delta_map(
                    null_mask,
                    orig_shape,
                    delta_scale_int,
                    orientation_map=corner_map,
                    base_mask=alpha_double_arr,
                    enable_circle_fill=False,
                )
            if delta_overlay_img is None:
                delta_overlay_img = self._render_alpha_delta_map(
                    null_mask,
                    orig_shape,
                    delta_scale_int,
                    orientation_map=corner_map,
                    base_mask=alpha_double_arr,
                    enable_circle_fill=False,
                )
            working_alpha = alpha_double

        if (double_w, double_h) != (dst_w, dst_h):
            scaled_alpha = working_alpha.resize((dst_w, dst_h), resample=Image.NEAREST)
        else:
            scaled_alpha = working_alpha

        base_factor = max(scale_x, scale_y, 2.0)
        up_factor = min(2, int(math.ceil(base_factor)))
        up_factor = max(1, up_factor)

        if up_factor > 1:
            hi_res = scaled_alpha.resize((dst_w * up_factor, dst_h * up_factor), resample=Image.NEAREST)
        else:
            hi_res = scaled_alpha.copy()

        smooth_strength = max(scale_x, scale_y, 1.0)
        # Preserve crisp triangle edges by avoiding Gaussian blur smoothing.
        blur_radius = 0.0
        if blur_radius > 0.0:
            hi_res = _safe_gaussian_blur(hi_res, blur_radius)

        if up_factor > 1:
            downsampled = hi_res.resize((dst_w, dst_h), resample=Image.LANCZOS)
        else:
            downsampled = hi_res
        edge_soften_radius = 0.0
        if edge_soften_radius > 0.0:
            downsampled = _safe_gaussian_blur(downsampled, edge_soften_radius)
        downsampled = _safe_unsharp_mask(downsampled, 1.1, percent=180, threshold=1)

        if delta_overlay_img is not None:
            if delta_overlay_img.size != (dst_w, dst_h):
                delta_overlay_img = delta_overlay_img.resize((dst_w, dst_h), resample=Resampling.BICUBIC)
            overlay_arr = np.asarray(delta_overlay_img, dtype=np.float32) / 255.0
            downsampled_arr = np.asarray(downsampled, dtype=np.float32) / 255.0
            if changed_mask_bool is not None:
                try:
                    changed_mask_img = Image.fromarray((changed_mask_bool.astype(np.uint8) * 255), "L")
                    if changed_mask_img.size != (dst_w, dst_h):
                        changed_mask_img = changed_mask_img.resize((dst_w, dst_h), resample=Resampling.NEAREST)
                    extension_mask = np.asarray(changed_mask_img, dtype=np.uint8) > 0
                except Exception:
                    extension_mask = overlay_arr > 0.5
            else:
                extension_mask = overlay_arr > 0.5
            extension_mask_final = extension_mask
            try:
                corner_img = Image.fromarray(np.asarray(corner_map, dtype=np.uint8), "L")
                if corner_img.size != (dst_w, dst_h):
                    corner_img = corner_img.resize((dst_w, dst_h), resample=Resampling.NEAREST)
                orientation_arr = np.asarray(corner_img, dtype=np.uint8)
                if extension_mask_final is not None:
                    orientation_arr = np.where(extension_mask_final, orientation_arr, 0).astype(np.uint8)
                orientation_final = orientation_arr
            except Exception:
                orientation_final = None
            circle_components = getattr(self, "_recent_circle_components", [])
            if circle_components:
                height, width = downsampled_arr.shape
                for comp in circle_components:
                    try:
                        cy_norm = float(comp["center_y_norm"])
                        cx_norm = float(comp["center_x_norm"])
                        ry_norm = float(comp["radius_y_norm"])
                        rx_norm = float(comp["radius_x_norm"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    cy = cy_norm * height
                    cx = cx_norm * width
                    ry = max(1e-6, ry_norm * height)
                    rx = max(1e-6, rx_norm * width)
                    y_min = max(0, int(math.floor(cy - ry - 1)))
                    y_max = min(height, int(math.ceil(cy + ry + 1)))
                    x_min = max(0, int(math.floor(cx - rx - 1)))
                    x_max = min(width, int(math.ceil(cx + rx + 1)))
                    if y_min >= y_max or x_min >= x_max:
                        continue
                    region_overlay = overlay_arr[y_min:y_max, x_min:x_max]
                    positive = region_overlay > 0.0
                    if not positive.any():
                        continue
                    region_down = downsampled_arr[y_min:y_max, x_min:x_max]
                    mask_reduce = positive & (region_down > region_overlay + 1e-3)
                    if mask_reduce.any():
                        region_down[mask_reduce] = region_overlay[mask_reduce]
                        downsampled_arr[y_min:y_max, x_min:x_max] = region_down
            combined = np.maximum(downsampled_arr, overlay_arr)
            downsampled = Image.fromarray((combined * 255.0 + 0.5).astype(np.uint8), "L")
            self._recent_circle_components = []

        if extension_mask_final is not None:
            try:
                extension_bool = np.asarray(extension_mask_final, dtype=bool)
                if extension_bool.shape != (dst_h, dst_w):
                    mask_img = Image.fromarray(extension_bool.astype(np.uint8) * 255, "L")
                    mask_img = mask_img.resize((dst_w, dst_h), resample=Resampling.NEAREST)
                    extension_bool = np.asarray(mask_img, dtype=np.uint8) > 0
                if changed_mask_bool is not None:
                    try:
                        changed_img = Image.fromarray((changed_mask_bool.astype(np.uint8) * 255), "L")
                        changed_img = changed_img.resize((dst_w, dst_h), resample=Resampling.NEAREST)
                        changed_mask_resized = np.asarray(changed_img, dtype=np.uint8) > 0
                        extension_bool = extension_bool & changed_mask_resized
                    except Exception:
                        pass
                self._recent_alpha_extension_mask = extension_bool
            except Exception:
                self._recent_alpha_extension_mask = None
            if orientation_final is not None:
                try:
                    orientation_arr = np.asarray(orientation_final, dtype=np.uint8)
                    if orientation_arr.shape != (dst_h, dst_w):
                        orient_img = Image.fromarray(orientation_arr, "L")
                        orient_img = orient_img.resize((dst_w, dst_h), resample=Resampling.NEAREST)
                        orientation_arr = np.asarray(orient_img, dtype=np.uint8)
                    if self._recent_alpha_extension_mask is not None:
                        orientation_arr = np.where(
                            self._recent_alpha_extension_mask,
                            orientation_arr,
                            0,
                        ).astype(np.uint8)
                    self._recent_alpha_extension_orientation = orientation_arr
                except Exception:
                    self._recent_alpha_extension_orientation = None
            triangle_defs = getattr(self, "_recent_triangle_components", [])
            triangles_final: List[Dict[str, Any]] = []
            if triangle_defs:
                for tri in triangle_defs:
                    try:
                        orientation_val = int(round(float(tri.get("orientation", 0.0))))
                        apex_x_norm = float(tri.get("apex_x_norm", 0.0))
                        apex_y_norm = float(tri.get("apex_y_norm", 0.0))
                        horiz_x_norm = float(tri.get("horizontal_x_norm", 0.0))
                        horiz_y_norm = float(tri.get("horizontal_y_norm", 0.0))
                        vert_x_norm = float(tri.get("vertical_x_norm", 0.0))
                        vert_y_norm = float(tri.get("vertical_y_norm", 0.0))
                    except (TypeError, ValueError):
                        continue
                    apex_x_norm = float(np.clip(apex_x_norm, 0.0, 1.0))
                    apex_y_norm = float(np.clip(apex_y_norm, 0.0, 1.0))
                    horiz_x_norm = float(np.clip(horiz_x_norm, 0.0, 1.0))
                    horiz_y_norm = float(np.clip(horiz_y_norm, 0.0, 1.0))
                    vert_x_norm = float(np.clip(vert_x_norm, 0.0, 1.0))
                    vert_y_norm = float(np.clip(vert_y_norm, 0.0, 1.0))
                    apex_x = apex_x_norm * float(dst_w)
                    apex_y = apex_y_norm * float(dst_h)
                    horiz_x = horiz_x_norm * float(dst_w)
                    horiz_y = horiz_y_norm * float(dst_h)
                    vert_x = vert_x_norm * float(dst_w)
                    vert_y = vert_y_norm * float(dst_h)
                    triangles_final.append(
                        {
                            "orientation": orientation_val,
                            "apex": (apex_y, apex_x),
                            "horizontal": (horiz_y, horiz_x),
                            "vertical": (vert_y, vert_x),
                        }
                    )
            self._recent_alpha_extension_triangles = triangles_final
            self._recent_triangle_components = []
            if triangle_fill_mask_result is not None and triangle_fill_mask_result.any():
                self._recent_edge_blur_radius = max(0.5, min(4.0, float(delta_scale_int)))
            else:
                self._recent_edge_blur_radius = 0.0
        else:
            self._recent_alpha_extension_mask = None
            self._recent_alpha_extension_orientation = None
            self._recent_alpha_extension_triangles = []
            self._recent_triangle_components = []
            self._recent_edge_blur_radius = 0.0

        arr = np.asarray(downsampled, dtype=np.float32) / 255.0
        arr = ((arr - 0.5) * 1.2) + 0.5
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "L")

    def _restore_original_alpha(
        self,
        source: Path,
        target: Path,
        *,
        edge_mode: str = "smooth",
        delta_scale: int = 1,
    ) -> None:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return

        alpha = self._load_scaled_alpha(
            source,
            self._image_size(target),
            delta_scale=delta_scale,
            edge_mode=edge_mode,
        )
        if alpha is None:
            return

        try:
            with Image.open(target) as result_img:
                result_rgba = result_img.convert("RGBA")
        except Exception:
            return

        result_rgba.putalpha(alpha)
        result_rgba = self._bleed_alpha_colors(result_rgba, alpha)
        result_rgba.save(target)

    @staticmethod
    def _image_size(path: Path) -> Tuple[int, int]:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return (0, 0)
        with Image.open(path) as img:
            return img.size

    @staticmethod
    def _safe_cleanup(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except PermissionError:
            logging.debug("Skipping cleanup for %s (permission error)", path)

def replace_file(source: Path, enhanced: Path) -> None:
    if source.resolve() == enhanced.resolve():
        return
    source.unlink()
    shutil.move(str(enhanced), str(source))


# --------- PK3 handling ----------


def collect_texture_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.strip().lower() for ext in extensions if ext.strip()}
    if "*" in normalized_exts:
        normalized_exts = {"*"}
    files: List[Path] = []
    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in normalized_exts:
            files.append(item)
        elif item.is_file() and "*" in normalized_exts:
            files.append(item)
    return files


def process_pk3(
    source: Path,
    dest: Path,
    upscaler: UpscaleExecutor,
    detail_scale: int,
    target_scale: int,
    max_pixels: int,
    texture_exts: Iterable[str],
    skip_categories: Collection[str],
    generate_sprite_materials: bool,
    disable_materials: bool,
    keep_temp: bool,
    dry_run: bool,
    output_mode: str,
    character_model_config: Optional[CharacterModelerConfig] = None,
) -> None:
    global Image
    temp_root = create_temp_dir(keep_temp)
    textures_dir = temp_root / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    character_collector: Optional[CharacterSpriteCollector] = None
    if character_model_config is not None and not dry_run:
        character_collector = CharacterSpriteCollector(temp_root / "pix2vox_sprites")

    normalized_exts = {ext.strip().lower() for ext in texture_exts if ext.strip()}
    use_all = "*" in normalized_exts
    skip_category_set = {category.lower() for category in skip_categories if category}
    entries: List[dict[str, Any]] = []
    material_records: Dict[Tuple[str, str], Dict[str, str]] = {}
    added_normal_files: set[str] = set()
    added_specular_files: set[str] = set()
    scaling_entries: Dict[str, Pk3ScaleEntry] = {}
    skipped_by_filter = 0
    diff_mode = output_mode == "diff"
    upscaled_outputs: Dict[str, bytes] = {}

    logging.info("Extracting %s", source)
    with zipfile.ZipFile(source, "r") as archive:
        infos = archive.infolist()
        material_skip_paths: Set[str] = set()
        material_skip_names: Set[str] = set()
        existing_material_refs: Dict[Tuple[str, str], Dict[str, str]] = {}
        for info in infos:
            upper_name = info.filename.upper()
            if upper_name != "GLDEFS":
                if not diff_mode and upper_name == "HIRESTEX":
                    try:
                        hirestex_bytes = archive.read(info)
                    except KeyError:
                        continue
                    parsed_entries = parse_hirestex_entries(hirestex_bytes)
                    scaling_entries.update(parsed_entries)
                continue
            try:
                gldefs_bytes = archive.read(info)
            except KeyError:
                continue
            paths, names, material_defs = _extract_material_map_references(gldefs_bytes)
            material_skip_paths.update(paths)
            material_skip_names.update(names)
            for key, value in material_defs.items():
                existing_entry = existing_material_refs.setdefault(key, {})
                existing_entry.update(value)

        def _is_locked_material_map(token: str) -> bool:
            normalized = token.replace("\\", "/").lower()
            if normalized in material_skip_paths:
                return True
            stem_token = PurePosixPath(normalized).stem.upper()
            return stem_token in material_skip_names

        def _should_delay_ui_entry(zinfo: zipfile.ZipInfo) -> bool:
            arcname_local = zinfo.filename
            if arcname_local.endswith("/"):
                return False
            category_guess = detect_texture_category(arcname_local)
            if category_guess != "ui":
                return False
            if use_all:
                return True
            suffix_local = Path(arcname_local).suffix.lower()
            if suffix_local and suffix_local in normalized_exts:
                return True
            if not suffix_local:
                return True
            return False

        delayed_ui_names: Set[str] = {info.filename for info in infos if _should_delay_ui_entry(info)}
        if delayed_ui_names:
            ordered_infos: List[zipfile.ZipInfo] = [info for info in infos if info.filename not in delayed_ui_names]
            ordered_infos.extend(info for info in infos if info.filename in delayed_ui_names)
        else:
            ordered_infos = list(infos)
        texture_total = _precount_pk3_textures(
            archive,
            ordered_infos,
            normalized_exts,
            use_all,
            max_pixels,
            skip_category_set,
            locked_paths=material_skip_paths,
            locked_names=material_skip_names,
        )
        processed = 0

        for info in ordered_infos:
            arcname = info.filename
            # Normalize to forward slashes for Zip
            arcname_posix = arcname.replace("\\", "/")
            if arcname.endswith("/"):
                if not diff_mode:
                    entries.append({"info": info, "data": None})
                continue
            if arcname in added_normal_files:
                continue

            data = archive.read(arcname)
            suffix = Path(arcname).suffix.lower()
            detected_ext = detect_image_extension(data)

            allowed_extension: Optional[str] = None
            if suffix and (use_all or suffix in normalized_exts):
                allowed_extension = suffix
            elif detected_ext and (use_all or detected_ext in normalized_exts):
                allowed_extension = detected_ext

            is_texture = not arcname.endswith("/") and allowed_extension is not None
            category = detect_texture_category(arcname) if is_texture else "other"

            if is_texture and (material_skip_paths or material_skip_names) and _is_locked_material_map(arcname_posix):
                logging.info("Skipping %s (referenced by GLDEFS material)", arcname)
                if not diff_mode:
                    entries.append({"info": info, "data": data})
                continue

            if is_texture and category in {"normal", "mask"}:
                if not diff_mode:
                    entries.append({"info": info, "data": data})
                continue

            if is_texture and skip_category_set and category in skip_category_set:
                skipped_by_filter += 1
                logging.info("Skipping %s (%s texture) per skip filter", arcname, category)
                if not diff_mode:
                    entries.append({"info": info, "data": data})
                continue

            material_kind_value: Optional[str] = None
            material_target_value: Optional[str] = None
            existing_refs: Optional[Dict[str, str]] = None
            if is_texture:
                material_kind_value = _determine_material_kind_pk3(category, arcname)
                material_target_value = _material_target_from_pk3(material_kind_value, arcname)
                lookup_target = (
                    material_target_value
                    if material_kind_value in {"sprite", "flat"}
                    else material_target_value.lower()
                )
                existing_refs = existing_material_refs.get((material_kind_value, lookup_target))

            counted = is_texture
            if is_texture:
                safe_name = hashlib.sha1(arcname.encode("utf-8")).hexdigest()
                extension = Path(arcname).suffix
                temp_extension = allowed_extension or extension
                payload_bytes = data
                convert_back: Optional[str] = None
                orig_offsets: Optional[Tuple[int, int]] = None
                orig_width = 0
                orig_height = 0

                source_extension = detected_ext or suffix
                if temp_extension is None or temp_extension == "":
                    temp_extension = source_extension or ".png"

                if source_extension == ".tga":
                    try:
                        from PIL import Image  # type: ignore
                    except ImportError as exc:  # pragma: no cover
                        raise RuntimeError(
                            "Pillow is required to convert TGA textures for upscaling. "
                            "Install with `python -m pip install pillow`."
                        ) from exc
                    with Image.open(io.BytesIO(data)) as img:
                        if img.mode not in ("RGB", "RGBA"):
                            img = img.convert("RGBA")
                        orig_width, orig_height = img.size
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        payload_bytes = buffer.getvalue()
                        temp_extension = ".png"
                        convert_back = "TGA"

                if temp_extension.lower() == ".png":
                    width, height = extract_png_dimensions(payload_bytes)
                    if width and height:
                        orig_width = width
                        orig_height = height
                    orig_offsets = extract_png_grab(payload_bytes)
                if orig_width == 0 or orig_height == 0:
                    orig_width = orig_width or 0
                    orig_height = orig_height or 0

                try:
                    with Image.open(io.BytesIO(payload_bytes)) as img_validate:
                        if orig_width == 0 or orig_height == 0:
                            orig_width, orig_height = img_validate.size
                        img_validate.load()
                except Exception as exc:
                    logging.warning("Skipping %s: unable to decode image (%s)", arcname, exc)
                    if not diff_mode:
                        entries.append({"info": info, "data": data})
                    if counted:
                        processed += 1
                        if texture_total > 0:
                            logging.info("%d of %d images processed", processed, texture_total)
                    continue

                skip_large_texture = False
                pixel_count = None
                if max_pixels > 0 and orig_width > 0 and orig_height > 0:
                    pixel_count = orig_width * orig_height
                    if pixel_count > max_pixels:
                        skip_large_texture = True
                        logging.info(
                            "Skipping %s (%dx%d = %d pixels exceeds limit %d)",
                            arcname,
                            orig_width,
                            orig_height,
                            pixel_count,
                            max_pixels,
                        )

                if skip_large_texture:
                    if not diff_mode:
                        entries.append({"info": info, "data": data})
                    if counted:
                        processed += 1
                        if texture_total > 0:
                            logging.info("%d of %d images processed", processed, texture_total)
                    continue

                temp_input = textures_dir / f"{safe_name}{temp_extension}"
                temp_input.parent.mkdir(parents=True, exist_ok=True)
                temp_input.write_bytes(payload_bytes)
                enhanced_path = temp_input.with_name(temp_input.stem + "_enhanced" + temp_extension)
                metadata = {
                    "source": arcname,
                    "original_width": orig_width,
                    "original_height": orig_height,
                    "detail_scale": detail_scale,
                    "target_scale": target_scale,
                    "keep_intermediate": keep_temp,
                }
                if keep_temp:
                    delta_path = temp_input.with_name(f"{temp_input.stem}_alpha_delta.png")
                    metadata["alpha_delta_path"] = str(delta_path)
                if character_collector is not None and category == "character":
                    character_collector.add_frame(
                        identifier=arcname,
                        image_bytes=payload_bytes,
                        extension=temp_extension,
                    )
                job = TextureJob(
                    input_path=temp_input,
                    output_path=enhanced_path,
                    detail_scale=detail_scale,
                    target_scale=target_scale,
                    dry_run=dry_run,
                    identifier=arcname,
                    category=category,
                    metadata=metadata,
                )
                result_path = upscaler.upscale(job)
                normal_bytes: Optional[bytes] = None
                specular_bytes: Optional[bytes] = None
                normal_temp_path: Optional[Path] = None
                if dry_run:
                    new_data = data
                else:
                    raw_result = result_path.read_bytes()
                    allow_materials = (
                        not disable_materials
                        and category not in {"normal", "mask"}
                        and category != "ui"
                        and (generate_sprite_materials or category not in MATERIAL_EXCLUDED_CATEGORIES)
                    )
                    if allow_materials:
                        normal_candidate = _build_normal_map_path(arcname).replace("\\", "/")
                        specular_candidate = _build_specular_map_path(arcname).replace("\\", "/")
                        normal_locked = bool(existing_refs and existing_refs.get("normal")) or _is_locked_material_map(
                            normal_candidate
                        )
                        spec_locked = bool(existing_refs and existing_refs.get("specular")) or _is_locked_material_map(
                            specular_candidate
                        )
                        if normal_locked:
                            normal_bytes = None
                        else:
                            normal_bytes = _generate_normal_map_bytes(raw_result)
                        if spec_locked:
                            specular_bytes = None
                        else:
                            specular_bytes = _generate_specular_map_bytes(raw_result)
                    else:
                        normal_bytes = None
                        specular_bytes = None
                    new_data = raw_result
                    new_width = 0
                    new_height = 0
                    scaled_offsets: Optional[Tuple[int, int]] = None
                    if convert_back == "TGA":
                        from PIL import Image  # type: ignore

                        with Image.open(io.BytesIO(new_data)) as img:
                            new_width, new_height = img.size
                            buffer = io.BytesIO()
                            img.save(buffer, format="TGA")
                            new_data = buffer.getvalue()
                    else:
                        if new_data.startswith(PNG_MAGIC):
                            width, height = extract_png_dimensions(new_data)
                            new_width = width
                            new_height = height
                        if not new_width or not new_height:
                            try:
                                with Image.open(io.BytesIO(new_data)) as img_out:
                                    new_width, new_height = img_out.size
                            except Exception:
                                new_width = new_width or 0
                                new_height = new_height or 0
                    if (
                        orig_offsets is not None
                        and orig_width
                        and orig_height
                        and new_width
                        and new_height
                    ):
                        scale_x = new_width / orig_width if orig_width else 1.0
                        scale_y = new_height / orig_height if orig_height else 1.0
                        scaled_offsets = (
                            int(round(orig_offsets[0] * scale_x)),
                            int(round(orig_offsets[1] * scale_y)),
                        )
                        if convert_back != "TGA" and new_data.startswith(PNG_MAGIC):
                            try:
                                new_data = set_png_grab_chunk(new_data, scaled_offsets)
                            except Exception:
                                pass
                    if normal_bytes and scaled_offsets is not None:
                        try:
                            normal_bytes = set_png_grab_chunk(normal_bytes, scaled_offsets)
                        except Exception:
                            pass
                    if specular_bytes and scaled_offsets is not None:
                        try:
                            specular_bytes = set_png_grab_chunk(specular_bytes, scaled_offsets)
                        except Exception:
                            pass
                    if keep_temp and normal_bytes:
                        normal_temp_path = enhanced_path.with_name(enhanced_path.stem + "_nm.png")
                        try:
                            normal_temp_path.write_bytes(normal_bytes)
                        except Exception as exc:
                            logging.debug("Failed to write temporary normal map %s: %s", normal_temp_path, exc)
                    if keep_temp and specular_bytes:
                        specular_temp_path = enhanced_path.with_name(enhanced_path.stem + "_sp.png")
                        try:
                            specular_temp_path.write_bytes(specular_bytes)
                        except Exception as exc:
                            logging.debug("Failed to write temporary specular map %s: %s", specular_temp_path, exc)
                    if (
                        category in {"sprite", "character", "ui"}
                        and orig_width
                        and orig_height
                        and new_width
                        and new_height
                        and (new_width != orig_width or new_height != orig_height)
                    ):
                        sprite_name = sanitize_lump_name(Path(arcname).stem).upper()
                        scaling_entries[sprite_name] = Pk3ScaleEntry(
                            name=sprite_name,
                            category=category,
                            original_width=orig_width,
                            original_height=orig_height,
                            new_width=new_width,
                            new_height=new_height,
                            path=arcname.replace("\\", "/"),
                            offset_x=int(scaled_offsets[0]) if scaled_offsets is not None else None,
                            offset_y=int(scaled_offsets[1]) if scaled_offsets is not None else None,
                        )

                    if not keep_temp:
                        try:
                            temp_input.unlink()
                        except FileNotFoundError:
                            pass
                        if result_path != temp_input:
                            try:
                                result_path.unlink()
                            except FileNotFoundError:
                                pass
                if diff_mode and not dry_run:
                    arcname_posix = arcname.replace("\\", "/")
                    upscaled_outputs[arcname_posix] = new_data
                elif not diff_mode:
                    entries.append({"info": info, "data": new_data})
                if not dry_run and normal_bytes:
                    normal_arcname = _build_normal_map_path(arcname).replace("\\", "/")
                    normal_token = normal_arcname
                    if material_kind_value is None or material_target_value is None:
                        material_kind_value = _determine_material_kind_pk3(category, arcname)
                        material_target_value = _material_target_from_pk3(material_kind_value, arcname)
                    record = material_records.setdefault(
                        (material_kind_value, material_target_value),
                        {},
                    )
                    record["normal"] = normal_token
                    if diff_mode:
                        upscaled_outputs[normal_arcname] = normal_bytes
                    else:
                        if normal_arcname not in added_normal_files:
                            normal_info = zipfile.ZipInfo(normal_arcname)
                            normal_info.compress_type = zipfile.ZIP_DEFLATED
                            entries.append({"info": normal_info, "data": normal_bytes})
                            added_normal_files.add(normal_arcname)
                        else:
                            for existing_entry in reversed(entries):
                                existing_info = existing_entry["info"]
                                if existing_info.filename == normal_arcname:
                                    existing_entry["data"] = normal_bytes
                                    break
                    if specular_bytes:
                        spec_arcname = _build_specular_map_path(arcname).replace("\\", "/")
                        record["specular"] = spec_arcname
                        if diff_mode:
                            upscaled_outputs[spec_arcname] = specular_bytes
                        else:
                            if spec_arcname not in added_specular_files:
                                spec_info = zipfile.ZipInfo(spec_arcname)
                                spec_info.compress_type = zipfile.ZIP_DEFLATED
                                entries.append({"info": spec_info, "data": specular_bytes})
                                added_specular_files.add(spec_arcname)
                            else:
                                for existing_entry in reversed(entries):
                                    existing_info = existing_entry["info"]
                                    if existing_info.filename == spec_arcname:
                                        existing_entry["data"] = specular_bytes
                                        break
                if counted:
                    processed += 1
                    if texture_total > 0:
                        logging.info("%d of %d images processed", processed, texture_total)
            else:
                if not diff_mode:
                    entries.append({"info": info, "data": data})

        if texture_total:
            logging.info("Processed %d of %d images", processed, texture_total)

    hirestex_payload = build_hirestex_payload(
        scaling_entries.values(),
        allowed_categories={"sprite", "character", "ui"},
    )
    if hirestex_payload and not diff_mode:
        entries = [
            entry
            for entry in entries
            if not isinstance(entry.get("info"), zipfile.ZipInfo)
            or entry["info"].filename.upper() != "HIRESTEX"
        ]
        info = zipfile.ZipInfo("HIRESTEX")
        info.compress_type = zipfile.ZIP_DEFLATED
        entries.append({"info": info, "data": hirestex_payload})

    material_block = _build_material_block(material_records)
    material_bytes = material_block.encode("ascii") if material_block else None
    if material_bytes and not diff_mode:
        appended = False
        for entry in entries:
            info = entry["info"]
            if isinstance(info, zipfile.ZipInfo) and info.filename.upper() == "GLDEFS":
                existing_data = entry.get("data")
                if existing_data is None:
                    existing = b""
                else:
                    existing = existing_data.rstrip(b"\r\n")
                if existing:
                    entry["data"] = existing + b"\n\n" + material_bytes
                else:
                    entry["data"] = material_bytes
                appended = True
                break
        if not appended:
            info = zipfile.ZipInfo("GLDEFS")
            info.compress_type = zipfile.ZIP_DEFLATED
            entries.append({"info": info, "data": material_bytes})

    if skip_category_set and skipped_by_filter:
        logging.info(
            "Skipped %d texture(s) in %s due to category filter.",
            skipped_by_filter,
            source.name,
        )

    if not diff_mode:
        dedup_map: Dict[str, dict[str, Any]] = {}
        deduped_entries: List[dict[str, Any]] = []
        for entry in reversed(entries):
            name = entry["info"].filename
            if name not in dedup_map:
                dedup_map[name] = entry
                deduped_entries.append(entry)
        entries = list(reversed(deduped_entries))

    if texture_total == 0:
        logging.warning("No textures with the selected extensions were found in %s", source)
    else:
        logging.info("Found %d texture(s) to upscale", texture_total)

    if character_collector is not None and character_collector.has_characters() and character_model_config is not None:
        run_character_modeler_pipeline(
            collector=character_collector,
            config=character_model_config,
            dry_run=dry_run,
        )

    if dry_run:
        cleanup_temp_dir(temp_root, keep_temp)
        logging.info("Dry run complete. No archive was written.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if diff_mode:
        diff_files: Dict[str, bytes] = dict(upscaled_outputs)
        if hirestex_payload:
            diff_files["HIRESTEX"] = hirestex_payload
        if material_bytes:
            diff_files["GLDEFS"] = material_bytes
        if not diff_files:
            logging.warning("No upscaled assets were generated; writing empty diff PK3.")
        logging.info("Writing diff PK3 -> %s", dest)
        with zipfile.ZipFile(dest, "w") as output_zip:
            for arcname in sorted(diff_files.keys()):
                data = diff_files[arcname]
                info = zipfile.ZipInfo(arcname)
                info.compress_type = zipfile.ZIP_DEFLATED
                output_zip.writestr(info, data)
        cleanup_temp_dir(temp_root, keep_temp)
        return

    logging.info("Writing enhanced PK3 -> %s", dest)
    with zipfile.ZipFile(dest, "w") as output_zip:
        for entry in entries:
            info: zipfile.ZipInfo = entry["info"]
            data = entry["data"]
            if data is None:
                output_zip.writestr(info, b"")
                continue

            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.comment = info.comment
            new_info.extra = info.extra
            new_info.internal_attr = info.internal_attr
            new_info.create_system = info.create_system
            new_info.create_version = info.create_version
            new_info.extract_version = info.extract_version
            new_info.flag_bits = info.flag_bits
            new_info.volume = info.volume
            output_zip.writestr(new_info, data)

    cleanup_temp_dir(temp_root, keep_temp)


# --------- WAD handling ----------


@dataclass
class WadLump:
    name: str
    data: bytes


@dataclass
class WadFile:
    magic: bytes
    lumps: List[WadLump]


@dataclass
class WadTextureEntry:
    index: int
    path: Path
    offsets: Optional[Tuple[int, int]]
    original_width: int
    original_height: int
    source_type: str
    lump_name: str
    disable_materials: bool = False


@dataclass
class CompositeTexturePatch:
    name: str
    origin_x: int
    origin_y: int


@dataclass
class CompositeTextureDef:
    name: str
    width: int
    height: int
    patches: List[CompositeTexturePatch]


@dataclass
class PatchReplacement:
    name: str
    original_width: int
    original_height: int
    new_width: int
    new_height: int
    data: bytes
    category: str
    hires_name: str
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None

    @property
    def scale_x(self) -> float:
        if self.original_width <= 0 or self.new_width <= 0:
            return 1.0
        return self.new_width / self.original_width

    @property
    def scale_y(self) -> float:
        if self.original_height <= 0 or self.new_height <= 0:
            return 1.0
        return self.new_height / self.original_height


def read_wad(path: Path) -> WadFile:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic not in (b"IWAD", b"PWAD"):
            raise ValueError(f"{path} is not a valid IWAD/PWAD file.")

        lump_count = struct.unpack("<I", handle.read(4))[0]
        directory_offset = struct.unpack("<I", handle.read(4))[0]

        directory: List[Tuple[int, int, str]] = []
        handle.seek(directory_offset)
        for _ in range(lump_count):
            lump_offset = struct.unpack("<I", handle.read(4))[0]
            lump_size = struct.unpack("<I", handle.read(4))[0]
            name_bytes = handle.read(8)
            name = name_bytes.rstrip(b"\x00").decode("ascii", errors="ignore")
            directory.append((lump_offset, lump_size, name))

        lumps: List[WadLump] = []
        for lump_offset, lump_size, name in directory:
            handle.seek(lump_offset)
            data = handle.read(lump_size)
            lumps.append(WadLump(name=name, data=data))

    return WadFile(magic=magic, lumps=lumps)


def write_wad(wad: WadFile, path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(wad.magic)
        handle.write(struct.pack("<I", len(wad.lumps)))
        directory_offset_placeholder_pos = handle.tell()
        handle.write(struct.pack("<I", 0))  # Placeholder for directory offset

        lump_offsets: List[int] = []
        for lump in wad.lumps:
            lump_offsets.append(handle.tell())
            handle.write(lump.data)

        directory_offset = handle.tell()
        for lump_offset, lump in zip(lump_offsets, wad.lumps):
            handle.write(struct.pack("<I", lump_offset))
            handle.write(struct.pack("<I", len(lump.data)))
            name_bytes = lump.name.encode("ascii", errors="ignore")[:8]
            padded = name_bytes.ljust(8, b"\x00")
            handle.write(padded)

        handle.seek(directory_offset_placeholder_pos)
        handle.write(struct.pack("<I", directory_offset))


def _is_probable_map_marker(name: str) -> bool:
    upper = name.upper()
    if len(upper) == 4 and upper.startswith("E") and upper[2] == "M" and upper[1].isdigit() and upper[3].isdigit():
        return True
    if len(upper) == 5 and upper.startswith("MAP") and upper[3:].isdigit():
        return True
    return False


def _extract_sidedef_texture_names(data: bytes) -> Set[str]:
    names: Set[str] = set()
    record_size = 30
    total = len(data)
    count = total // record_size
    for idx in range(count):
        base = idx * record_size
        for offset in (4, 12, 20):
            start = base + offset
            end = start + 8
            if end > total:
                continue
            raw = data[start:end]
            text = raw.rstrip(b"\x00").decode("ascii", errors="ignore").strip().upper()
            if text and text != "-":
                names.add(text)
    return names


def _extract_sector_flat_names(data: bytes) -> Set[str]:
    names: Set[str] = set()
    record_size = 26
    total = len(data)
    count = total // record_size
    for idx in range(count):
        base = idx * record_size
        for offset in (4, 12):
            start = base + offset
            end = start + 8
            if end > total:
                continue
            raw = data[start:end]
            text = raw.rstrip(b"\x00").decode("ascii", errors="ignore").strip().upper()
            if text and text != "-":
                names.add(text)
    return names


def detect_map_texture_usage(wad: "WadFile") -> Tuple[Set[str], Set[str]]:
    used_textures: Set[str] = set()
    used_flats: Set[str] = set()
    lumps = wad.lumps
    total = len(lumps)
    for index, lump in enumerate(lumps):
        name = lump.name.upper()
        if not _is_probable_map_marker(name):
            continue
        lookahead = index + 1
        while lookahead < total:
            marker = lumps[lookahead].name.upper()
            if _is_probable_map_marker(marker):
                break
            if marker == "SIDEDEFS":
                used_textures.update(_extract_sidedef_texture_names(lumps[lookahead].data))
            elif marker == "SECTORS":
                used_flats.update(_extract_sector_flat_names(lumps[lookahead].data))
            lookahead += 1
    return used_textures, used_flats


def map_texture_to_patches(
    texture_names: Collection[str],
    composite_defs: Mapping[str, "CompositeTextureDef"],
) -> Set[str]:
    mapped: Set[str] = set()
    for name in texture_names:
        key = name.upper()
        composite = composite_defs.get(key)
        if composite:
            for patch in composite.patches:
                mapped.add(patch.name.upper())
        else:
            mapped.add(key)
    return mapped


def collect_wad_textures(
    wad: WadFile,
    textures_dir: Path,
    palette: Sequence[Tuple[int, int, int]],
    used_textures: Optional[Collection[str]] = None,
    used_flats: Optional[Collection[str]] = None,
    used_patches: Optional[Collection[str]] = None,
) -> List[WadTextureEntry]:
    textures_dir.mkdir(parents=True, exist_ok=True)
    collected: List[WadTextureEntry] = []
    section_stack: List[str] = []
    used_texture_set = {value.upper() for value in used_textures or [] if value}
    used_flat_set = {value.upper() for value in used_flats or [] if value}
    used_patch_set = {value.upper() for value in used_patches or [] if value}

    def should_disable(name: str, source_type: str, category_hint: str) -> bool:
        if category_hint == "ui":
            return True
        upper_name = name.upper()
        if source_type == "flat":
            if used_flat_set and upper_name not in used_flat_set:
                return True
        elif source_type in {"patch", "image"}:
            if used_patch_set and upper_name not in used_patch_set and upper_name not in used_texture_set:
                return True
        return False

    for index, lump in enumerate(wad.lumps):
        name = lump.name.upper()
        if name.endswith("_START"):
            section = classify_marker(name)
            if section:
                section_stack.append(section)
            continue
        if name.endswith("_END"):
            section = classify_marker(name)
            if section and section in section_stack:
                # Remove the last matching section
                for pos in range(len(section_stack) - 1, -1, -1):
                    if section_stack[pos] == section:
                        section_stack.pop(pos)
                        break
            continue

        if not lump.data:
            continue

        current_section = section_stack[-1] if section_stack else None
        ext = detect_image_extension(lump.data)
        sanitized = sanitize_lump_name(lump.name)

        if ext:
            filename = f"{index:04d}_{sanitized}{ext}"
            file_path = textures_dir / filename
            file_path.write_bytes(lump.data)
            width, height = extract_png_dimensions(lump.data)
            offsets = extract_png_grab(lump.data)
            inferred_source = "image"
            if current_section in {"flat", "patch", "sprite"}:
                inferred_source = current_section
            category_hint = detect_texture_category(lump.name, inferred_source if inferred_source in {"patch", "sprite", "flat"} else "image")
            disable_materials = should_disable(lump.name, inferred_source, category_hint)
            collected.append(
                WadTextureEntry(
                    index=index,
                    path=file_path,
                    offsets=offsets,
                    original_width=width,
                    original_height=height,
                    source_type=inferred_source,
                    lump_name=lump.name,
                    disable_materials=disable_materials,
                )
            )
            continue

        if current_section == "hires":
            continue
        if current_section == "flat":
            decoded = decode_flat(lump.data, palette)
            if not decoded:
                continue
            filename = f"{index:04d}_{sanitized}.png"
            file_path = textures_dir / filename
            write_png_rgba(file_path, decoded)
            category_hint = detect_texture_category(lump.name, "flat")
            disable_materials = should_disable(lump.name, "flat", category_hint)
            collected.append(
                WadTextureEntry(
                    index=index,
                    path=file_path,
                    offsets=None,
                    original_width=decoded.width,
                    original_height=decoded.height,
                    source_type="flat",
                    lump_name=lump.name,
                    disable_materials=disable_materials,
                )
            )
            continue

        if current_section in {"patch", "sprite"}:
            decoded = decode_patch(lump.data, palette)
            if not decoded:
                continue
            filename = f"{index:04d}_{sanitized}.png"
            file_path = textures_dir / filename
            write_png_rgba(file_path, decoded)
            source_type = "sprite" if current_section == "sprite" else "patch"
            category_hint = detect_texture_category(lump.name, source_type)
            disable_materials = should_disable(lump.name, source_type, category_hint)
            collected.append(
                WadTextureEntry(
                    index=index,
                    path=file_path,
                    offsets=(decoded.left_offset, decoded.top_offset),
                    original_width=decoded.width,
                    original_height=decoded.height,
                    source_type=source_type,
                    lump_name=lump.name,
                    disable_materials=disable_materials,
                )
            )
            continue

        # Marker-less patch lumps (e.g., Doom UI assets) still use the classic patch format.
        # As a fallback, try decoding the lump as a patch when no section classification applies.
        decoded_fallback = decode_patch(lump.data, palette)
        if decoded_fallback:
            fallback_category = detect_texture_category(lump.name, "patch")
            filename = f"{index:04d}_{sanitized}.png"
            file_path = textures_dir / filename
            write_png_rgba(file_path, decoded_fallback)
            disable_materials = should_disable(lump.name, "patch", fallback_category)
            collected.append(
                WadTextureEntry(
                    index=index,
                    path=file_path,
                    offsets=(decoded_fallback.left_offset, decoded_fallback.top_offset),
                    original_width=decoded_fallback.width,
                    original_height=decoded_fallback.height,
                    source_type="patch",
                    lump_name=lump.name,
                    disable_materials=disable_materials,
                )
            )

    return collected


def read_wad_pnames(wad: WadFile) -> List[str]:
    for lump in wad.lumps:
        if lump.name.upper() == "PNAMES":
            data = lump.data
            if len(data) < 4:
                return []
            patch_count = struct.unpack_from("<I", data, 0)[0]
            names: List[str] = []
            offset = 4
            for idx in range(patch_count):
                start = offset + idx * 8
                end = start + 8
                if end > len(data):
                    break
                raw = data[start:end]
                names.append(raw.rstrip(b"\x00").decode("ascii", errors="ignore").upper())
            return names
    return []


def parse_texture_lump(data: bytes, patch_names: Sequence[str]) -> List[CompositeTextureDef]:
    if len(data) < 4:
        return []
    texture_count = struct.unpack_from("<I", data, 0)[0]
    required = 4 + texture_count * 4
    if len(data) < required:
        return []
    offsets = struct.unpack_from("<" + "I" * texture_count, data, 4)
    textures: List[CompositeTextureDef] = []
    for offset in offsets:
        if offset + 22 > len(data):
            continue
        name = data[offset : offset + 8].rstrip(b"\x00").decode("ascii", errors="ignore").upper()
        width = struct.unpack_from("<H", data, offset + 12)[0]
        height = struct.unpack_from("<H", data, offset + 14)[0]
        patch_count = struct.unpack_from("<H", data, offset + 20)[0]
        pos = offset + 22
        patches: List[CompositeTexturePatch] = []
        for _ in range(patch_count):
            if pos + 10 > len(data):
                break
            origin_x, origin_y, patch_index, stepdir, colormap = struct.unpack_from("<hhHHH", data, pos)
            pos += 10
            if patch_index >= len(patch_names):
                continue
            patch_name = patch_names[patch_index]
            if not patch_name:
                continue
            patches.append(CompositeTexturePatch(name=patch_name, origin_x=origin_x, origin_y=origin_y))
        if patches:
            textures.append(CompositeTextureDef(name=name, width=width, height=height, patches=patches))
    return textures


def collect_composite_textures(wad: WadFile) -> Dict[str, CompositeTextureDef]:
    patch_names = read_wad_pnames(wad)
    if not patch_names:
        return {}
    texture_defs: Dict[str, CompositeTextureDef] = {}
    for lump in wad.lumps:
        name = lump.name.upper()
        if name.startswith("TEXTURE"):
            for tex in parse_texture_lump(lump.data, patch_names):
                texture_defs.setdefault(tex.name, tex)
    return texture_defs


def _resample_filter() -> Any:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    return Image.LANCZOS  # type: ignore[attr-defined]


def compose_texture_image(
    texture: CompositeTextureDef,
    patch_images: Dict[str, bytes],
    patch_replacements: Dict[str, PatchReplacement],
    patch_sizes: Dict[str, Tuple[int, int]],
    palette: Sequence[Tuple[int, int, int]],
    lump_lookup: Dict[str, WadLump],
) -> Optional[bytes]:
    if not any(p.name in patch_replacements for p in texture.patches):
        return None

    resample = _resample_filter()
    placements: List[Tuple[str, Image.Image, Tuple[int, int]]] = []
    scale_x_values: List[float] = []
    scale_y_values: List[float] = []

    for ref in texture.patches:
        patch_name = ref.name
        patch_bytes = patch_images.get(patch_name)
        if patch_bytes is not None:
            image = Image.open(io.BytesIO(patch_bytes))
        else:
            lump = lump_lookup.get(patch_name)
            if not lump:
                continue
            decoded = decode_patch(lump.data, palette)
            if not decoded:
                continue
            image = Image.frombytes("RGBA", (decoded.width, decoded.height), decoded.rgba)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        orig_width, orig_height = patch_sizes.get(patch_name, (image.width, image.height))
        if orig_width <= 0 or orig_height <= 0:
            orig_width, orig_height = image.width, image.height
        info = patch_replacements.get(patch_name)
        if info:
            scale_x_values.append(info.scale_x)
            scale_y_values.append(info.scale_y)
        else:
            if orig_width:
                scale_x_values.append(image.width / orig_width)
            if orig_height:
                scale_y_values.append(image.height / orig_height)
        placements.append((patch_name, image, (orig_width, orig_height, ref.origin_x, ref.origin_y)))

    if not placements:
        return None

    scale_x = max(scale_x_values) if scale_x_values else 1.0
    scale_y = max(scale_y_values) if scale_y_values else 1.0
    if scale_x <= 0 or scale_y <= 0:
        return None

    canvas_width = max(1, int(round(texture.width * scale_x)))
    canvas_height = max(1, int(round(texture.height * scale_y)))

    prepared: List[Tuple[Image.Image, int, int]] = []
    min_x = 0
    min_y = 0
    max_x = canvas_width
    max_y = canvas_height

    for patch_name, image, (orig_width, orig_height, origin_x, origin_y) in placements:
        target_width = max(1, int(round(orig_width * scale_x)))
        target_height = max(1, int(round(orig_height * scale_y)))
        if image.width != target_width or image.height != target_height:
            image = image.resize((target_width, target_height), resample)
        dest_x = int(round(origin_x * scale_x))
        dest_y = int(round(origin_y * scale_y))
        min_x = min(min_x, dest_x)
        min_y = min(min_y, dest_y)
        max_x = max(max_x, dest_x + image.width)
        max_y = max(max_y, dest_y + image.height)
        prepared.append((image, dest_x, dest_y))

    offset_x = -min(0, min_x)
    offset_y = -min(0, min_y)
    canvas_total_width = max(canvas_width + offset_x, max_x + offset_x)
    canvas_total_height = max(canvas_height + offset_y, max_y + offset_y)
    canvas = Image.new("RGBA", (canvas_total_width, canvas_total_height), (0, 0, 0, 0))

    for image, dest_x, dest_y in prepared:
        canvas.paste(image, (dest_x + offset_x, dest_y + offset_y), image)

    crop_left = offset_x
    crop_top = offset_y
    crop_right = crop_left + canvas_width
    crop_bottom = crop_top + canvas_height
    cropped = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def build_composite_hires_payloads(
    wad: WadFile,
    patch_images: Dict[str, bytes],
    patch_replacements: Dict[str, PatchReplacement],
    patch_sizes: Dict[str, Tuple[int, int]],
    palette: Sequence[Tuple[int, int, int]],
    composite_textures: Optional[Dict[str, CompositeTextureDef]] = None,
) -> Tuple[
    List[Tuple[str, bytes]],
    Dict[str, Tuple[float, float]],
    Dict[str, Tuple[int, int]],
]:
    if not patch_replacements:
        return [], {}, {}
    if composite_textures is None:
        composite_textures = collect_composite_textures(wad)
    if not composite_textures:
        return [], {}, {}
    lump_lookup: Dict[str, WadLump] = {}
    for lump in wad.lumps:
        lump_lookup.setdefault(lump.name.upper(), lump)
    payloads: List[Tuple[str, bytes]] = []
    scale_defs: Dict[str, Tuple[float, float]] = {}
    base_sizes: Dict[str, Tuple[int, int]] = {}
    for name, texture in composite_textures.items():
        payload = compose_texture_image(texture, patch_images, patch_replacements, patch_sizes, palette, lump_lookup)
        if payload:
            lump_name = name.upper()[:8]
            payloads.append((lump_name, payload))
            width, height = extract_png_dimensions(payload)
            if width and height and texture.width and texture.height:
                scale_x = width / texture.width
                scale_y = height / texture.height
                scale_defs[lump_name] = (scale_x, scale_y)
            if texture.width and texture.height:
                base_sizes[lump_name] = (texture.width, texture.height)
    return payloads, scale_defs, base_sizes


def process_wad(
    source: Path,
    dest: Path,
    upscaler: UpscaleExecutor,
    detail_scale: int,
    target_scale: int,
    max_pixels: int,
    skip_categories: Collection[str],
    generate_sprite_materials: bool,
    disable_materials: bool,
    keep_temp: bool,
    dry_run: bool,
    output_mode: str,
    character_model_config: Optional[CharacterModelerConfig] = None,
) -> None:
    temp_root = create_temp_dir(keep_temp)
    wad = read_wad(source)
    material_skip_names: Set[str] = set()
    existing_material_refs: Dict[Tuple[str, str], Dict[str, str]] = {}
    for lump in wad.lumps:
        if lump.name.upper() != "GLDEFS" or not lump.data:
            continue
        _, names, material_defs = _extract_material_map_references(lump.data)
        material_skip_names.update(names)
        for key, value in material_defs.items():
            existing_entry = existing_material_refs.setdefault(key, {})
            existing_entry.update(value)

    def _is_locked_material_reference(token: str) -> bool:
        normalized = token.replace("\\", "/")
        stem_token = PurePosixPath(normalized).stem.upper()
        if stem_token and stem_token in material_skip_names:
            return True
        return normalized.upper() in material_skip_names

    skip_category_set = {category.lower() for category in skip_categories if category}

    textures_dir = temp_root / "wad_textures"
    palette = extract_wad_palette(wad)
    composite_textures = collect_composite_textures(wad)
    used_texture_names, used_flat_names = detect_map_texture_usage(wad)
    used_patch_names = map_texture_to_patches(used_texture_names, composite_textures) if used_texture_names else set()
    textures = collect_wad_textures(
        wad,
        textures_dir,
        palette,
        used_textures=used_texture_names,
        used_flats=used_flat_names,
        used_patches=used_patch_names,
    )
    total_detected = len(textures)
    if total_detected == 0:
        logging.warning("No texture lumps were detected in %s", source)
    else:
        logging.info("Detected %d texture lump(s) for analysis", total_detected)

    filtered_entries: List[Tuple[WadTextureEntry, str]] = []
    skipped_by_filter = 0
    material_skipped = 0
    for entry in textures:
        category = detect_texture_category(entry.lump_name, entry.source_type)
        if entry.lump_name.upper() in material_skip_names:
            logging.info("Skipping %s (referenced by GLDEFS material)", entry.lump_name)
            material_skipped += 1
            continue
        if skip_category_set and category in skip_category_set:
            skipped_by_filter += 1
            logging.info("Skipping %s (%s texture) per skip filter", entry.lump_name, category)
            continue
        filtered_entries.append((entry, category))

    remaining_after_filters = len(filtered_entries)
    if total_detected and remaining_after_filters:
        if material_skipped or skipped_by_filter:
            logging.info(
                "Will process %d texture lump(s) after excluding %d GLDEFS-locked and %d filter-skipped.",
                remaining_after_filters,
                material_skipped,
                skipped_by_filter,
            )
        else:
            logging.info("Will process %d texture lump(s).", remaining_after_filters)
    elif total_detected and not remaining_after_filters:
        logging.info(
            "All %d detected texture lump(s) were excluded (%d GLDEFS-locked, %d filter-skipped).",
            total_detected,
            material_skipped,
            skipped_by_filter,
        )

    if skip_category_set:
        if textures and skipped_by_filter and filtered_entries:
            logging.info(
                "Processing %d texture lump(s) after skipping %d via category filter.",
                len(filtered_entries),
                skipped_by_filter,
            )
        elif textures and not filtered_entries:
            logging.info("No textures remain to upscale after applying category filters.")

    if filtered_entries:
        non_ui_entries = [item for item in filtered_entries if item[1] != "ui"]
        ui_entries = [item for item in filtered_entries if item[1] == "ui"]
        filtered_entries = non_ui_entries + ui_entries

    hires_payloads: List[Tuple[str, bytes]] = []
    material_records: Dict[Tuple[str, str], Dict[str, str]] = {}
    patch_images: Dict[str, bytes] = {}
    patch_replacements: Dict[str, PatchReplacement] = {}
    patch_sizes: Dict[str, Tuple[int, int]] = {}
    total_textures = len(filtered_entries)
    processed = 0
    existing_hires_names: set[str] = {lump.name.upper() for lump in wad.lumps}
    diff_mode = output_mode == "diff"
    diff_outputs: Dict[str, bytes] = {}
    diff_entry_map: Dict[str, Pk3ScaleEntry] = {}
    diff_normal_paths: set[str] = set()
    diff_spec_paths: set[str] = set()
    character_collector: Optional[CharacterSpriteCollector] = None
    if character_model_config is not None and not dry_run:
        character_collector = CharacterSpriteCollector(temp_root / "pix2vox_sprites")

    for entry, category in filtered_entries:
        skip_large_texture = False
        pixel_count = None
        lump_token = entry.lump_name.upper()[:8]
        if max_pixels > 0 and entry.original_width > 0 and entry.original_height > 0:
            pixel_count = entry.original_width * entry.original_height
            if pixel_count > max_pixels:
                skip_large_texture = True
                logging.info(
                    "Skipping %s (%dx%d = %d pixels exceeds limit %d)",
                    entry.lump_name,
                    entry.original_width,
                    entry.original_height,
                    pixel_count,
                    max_pixels,
                )
        if skip_large_texture:
            if total_textures:
                processed += 1
                logging.info("%d of %d images processed", processed, total_textures)
            continue

        material_kind_value = _determine_material_kind_wad(entry.source_type, category)
        material_target_value = _material_target_from_wad(entry.lump_name)
        lookup_target = (
            material_target_value
            if material_kind_value in {"sprite", "flat"}
            else material_target_value.lower()
        )
        existing_refs = existing_material_refs.get((material_kind_value, lookup_target))

        enhanced_path = entry.path.with_name(entry.path.stem + "_enhanced" + entry.path.suffix)
        metadata = {
            "source": entry.lump_name,
            "source_type": entry.source_type,
            "original_width": entry.original_width,
            "original_height": entry.original_height,
            "detail_scale": detail_scale,
            "target_scale": target_scale,
            "keep_intermediate": keep_temp,
        }
        if keep_temp:
            delta_path = entry.path.with_name(f"{entry.path.stem}_alpha_delta.png")
            metadata["alpha_delta_path"] = str(delta_path)
        if character_collector is not None and category == "character":
            try:
                staged_bytes = entry.path.read_bytes()
            except Exception as exc:
                logging.debug("Unable to stage %s for Pix2Vox+: %s", entry.lump_name, exc)
            else:
                character_collector.add_frame(
                    identifier=entry.lump_name,
                    image_bytes=staged_bytes,
                    extension=entry.path.suffix,
                )
        job = TextureJob(
            input_path=entry.path,
            output_path=enhanced_path,
            detail_scale=detail_scale,
            target_scale=target_scale,
            dry_run=dry_run,
            identifier=entry.lump_name,
            category=category,
            metadata=metadata,
        )
        result_path = upscaler.upscale(job)
        if total_textures:
            processed += 1
            logging.info("%d of %d images processed", processed, total_textures)
        if dry_run:
            continue
        result_bytes = result_path.read_bytes()
        diff_arcname: Optional[str] = None
        if diff_mode:
            suffix = entry.path.suffix if entry.path.suffix else ".png"
            diff_arcname = build_wad_diff_arcname(entry.lump_name, category, entry.source_type, suffix)
            diff_outputs[diff_arcname] = result_bytes
        new_width: Optional[int] = None
        new_height: Optional[int] = None
        normal_bytes: Optional[bytes] = None
        specular_bytes: Optional[bytes] = None
        normal_temp_path: Optional[Path] = None
        specular_temp_path: Optional[Path] = None
        try:
            new_width, new_height = extract_png_dimensions(result_bytes)
        except ValueError:
            new_width = None
            new_height = None
        allow_materials = (
            not disable_materials
            and not entry.disable_materials
            and category not in {"normal", "mask"}
            and category != "ui"
            and (generate_sprite_materials or category not in MATERIAL_EXCLUDED_CATEGORIES)
        )
        normal_locked = bool(existing_refs and existing_refs.get("normal"))
        spec_locked = bool(existing_refs and existing_refs.get("specular"))
        if allow_materials:
            if not normal_locked:
                normal_bytes = _generate_normal_map_bytes(result_bytes)
                if keep_temp and normal_bytes:
                    normal_temp_path = enhanced_path.with_name(enhanced_path.stem + "_nm.png")
                    try:
                        normal_temp_path.write_bytes(normal_bytes)
                    except Exception as exc:
                        logging.debug("Failed to write temporary normal map %s: %s", normal_temp_path, exc)
            if not spec_locked:
                specular_bytes = _generate_specular_map_bytes(result_bytes)
                if keep_temp and specular_bytes:
                    specular_temp_path = enhanced_path.with_name(enhanced_path.stem + "_sp.png")
                    try:
                        specular_temp_path.write_bytes(specular_bytes)
                    except Exception as exc:
                        logging.debug("Failed to write temporary specular map %s: %s", specular_temp_path, exc)
        if not dry_run and not keep_temp:
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass
        if entry.source_type in {"patch", "flat", "sprite"}:
            patch_name = entry.lump_name.upper()[:8]
            existing_hires_names.add(patch_name.upper())
            adjusted_offsets: Optional[Tuple[int, int]] = None
            if entry.offsets is not None:
                adjusted_offsets = entry.offsets
                if (
                    category in {"sprite", "character", "ui"}
                    and entry.original_width > 0
                    and entry.original_height > 0
                    and new_width
                    and new_height
                ):
                    scale_x = new_width / entry.original_width
                    scale_y = new_height / entry.original_height
                    lump_upper = entry.lump_name.upper()
                    new_left = int(round(entry.offsets[0] * scale_x))
                    if lump_upper in {"BAR1A0", "BAR1B0"}:
                        bottom_depth = entry.original_height - entry.offsets[1]
                        new_top = int(round(new_height - bottom_depth))
                    else:
                        new_top = int(round(entry.offsets[1] * scale_y))
                    adjusted_offsets = (new_left, new_top)
                adjusted_offsets = (int(adjusted_offsets[0]), int(adjusted_offsets[1]))
                try:
                    result_bytes = set_png_grab_chunk(result_bytes, adjusted_offsets)
                except Exception:
                    pass
                if normal_bytes:
                    try:
                        normal_bytes = set_png_grab_chunk(normal_bytes, adjusted_offsets)
                    except Exception:
                        pass
                    if keep_temp and normal_temp_path:
                        try:
                            normal_temp_path.write_bytes(normal_bytes)
                        except Exception as exc:
                            logging.debug("Failed to update temporary normal map %s: %s", normal_temp_path, exc)
                if specular_bytes:
                    try:
                        specular_bytes = set_png_grab_chunk(specular_bytes, adjusted_offsets)
                    except Exception:
                        pass
            wad.lumps[entry.index].data = result_bytes
            patch_images[patch_name] = result_bytes
            patch_sizes[patch_name] = (entry.original_width, entry.original_height)
            if entry.source_type in {"patch", "sprite"} and entry.original_width > 0 and entry.original_height > 0:
                current_width = new_width or entry.original_width
                current_height = new_height or entry.original_height
                if current_width and current_height and (
                    current_width != entry.original_width or current_height != entry.original_height
                ):
                    patch_replacements[patch_name] = PatchReplacement(
                        name=patch_name,
                        original_width=entry.original_width,
                        original_height=entry.original_height,
                        new_width=current_width,
                        new_height=current_height,
                        data=result_bytes,
                        category=category,
                        hires_name=patch_name,
                        offset_x=adjusted_offsets[0] if adjusted_offsets is not None else None,
                        offset_y=adjusted_offsets[1] if adjusted_offsets is not None else None,
                    )
        else:
            wad.lumps[entry.index].data = result_bytes
        if diff_mode and diff_arcname is not None and category not in {"normal", "mask"}:
            final_width = new_width or entry.original_width
            final_height = new_height or entry.original_height
            if final_width and final_height and entry.source_type != "flat":
                diff_entry_map[lump_token] = Pk3ScaleEntry(
                    name=lump_token,
                    category=category,
                    original_width=entry.original_width,
                    original_height=entry.original_height,
                    new_width=final_width,
                    new_height=final_height,
                    path=diff_arcname,
                    offset_x=int(adjusted_offsets[0]) if adjusted_offsets is not None else None,
                    offset_y=int(adjusted_offsets[1]) if adjusted_offsets is not None else None,
                )
        if diff_mode and diff_arcname is not None:
            diff_outputs[diff_arcname] = result_bytes
        if normal_bytes:
            if material_kind_value is None or material_target_value is None:
                material_kind_value = _determine_material_kind_wad(entry.source_type, category)
                material_target_value = _material_target_from_wad(entry.lump_name)
                lookup_target = (
                    material_target_value
                    if material_kind_value in {"sprite", "flat"}
                    else material_target_value.lower()
                )
                existing_refs = existing_material_refs.get((material_kind_value, lookup_target))
                normal_locked = bool(existing_refs and existing_refs.get("normal"))
                spec_locked = bool(existing_refs and existing_refs.get("specular"))
            record = material_records.setdefault((material_kind_value, material_target_value), {})
            if diff_mode:
                if diff_arcname is None:
                    suffix = entry.path.suffix if entry.path.suffix else ".png"
                    diff_arcname = build_wad_diff_arcname(entry.lump_name, category, entry.source_type, suffix)
                    diff_outputs[diff_arcname] = result_bytes
                normal_arcname = _build_normal_map_path(diff_arcname).replace("\\", "/")
                if not _is_locked_material_reference(normal_arcname) and normal_arcname not in diff_normal_paths:
                    diff_outputs[normal_arcname] = normal_bytes
                    diff_normal_paths.add(normal_arcname)
                    record["normal"] = normal_arcname
                if specular_bytes:
                    spec_arcname = _build_specular_map_path(diff_arcname).replace("\\", "/")
                    if (
                        not _is_locked_material_reference(spec_arcname)
                        and spec_arcname not in diff_spec_paths
                    ):
                        diff_outputs[spec_arcname] = specular_bytes
                        diff_spec_paths.add(spec_arcname)
                        record["specular"] = spec_arcname
            else:
                if "normal" not in record:
                    normal_lump_name = _generate_normal_lump_name(entry.lump_name, existing_hires_names)
                    if not _is_locked_material_reference(normal_lump_name):
                        hires_payloads.append((normal_lump_name, normal_bytes))
                        record["normal"] = normal_lump_name
                if specular_bytes and "specular" not in record:
                    spec_lump_name = _generate_specular_lump_name(entry.lump_name, existing_hires_names)
                    if not _is_locked_material_reference(spec_lump_name):
                        hires_payloads.append((spec_lump_name, specular_bytes))
                        record["specular"] = spec_lump_name

    if not dry_run:
        composite_payloads, composite_scales, composite_base_sizes = build_composite_hires_payloads(
            wad=wad,
            patch_images=patch_images,
            patch_replacements=patch_replacements,
            patch_sizes=patch_sizes,
            palette=palette,
            composite_textures=composite_textures,
        )
        if composite_payloads:
            for composite_name, composite_bytes in composite_payloads:
                material_kind = "texture"
                material_target = _material_target_from_wad(composite_name)
                lookup_target = material_target.lower()
                existing_refs_comp = existing_material_refs.get((material_kind, lookup_target))
                normal_locked = bool(existing_refs_comp and existing_refs_comp.get("normal"))
                spec_locked = bool(existing_refs_comp and existing_refs_comp.get("specular"))
                normal_bytes = _generate_normal_map_bytes(composite_bytes) if not normal_locked else None
                specular_bytes = _generate_specular_map_bytes(composite_bytes) if not spec_locked else None
                if diff_mode:
                    sanitized_name = sanitize_lump_name(composite_name).upper()
                    arcname = str(PurePosixPath("textures") / f"{sanitized_name}.png")
                    diff_outputs[arcname] = composite_bytes
                    base_width, base_height = composite_base_sizes.get(composite_name, (0, 0))
                    new_width, new_height = extract_png_dimensions(composite_bytes)
                    if not new_width or not new_height:
                        new_width = new_width or base_width
                        new_height = new_height or base_height
                    original_width = base_width or new_width
                    original_height = base_height or new_height
                    if original_width and original_height and new_width and new_height:
                        diff_entry_map[sanitized_name] = Pk3ScaleEntry(
                            name=sanitized_name,
                            category="world",
                            original_width=original_width,
                            original_height=original_height,
                            new_width=new_width,
                            new_height=new_height,
                            path=arcname,
                        )
                    record = material_records.setdefault((material_kind, material_target), {})
                    if normal_bytes:
                        normal_arcname = _build_normal_map_path(arcname).replace("\\", "/")
                        if not _is_locked_material_reference(normal_arcname) and normal_arcname not in diff_normal_paths:
                            diff_outputs[normal_arcname] = normal_bytes
                            diff_normal_paths.add(normal_arcname)
                            record["normal"] = normal_arcname
                    if specular_bytes:
                        spec_arcname = _build_specular_map_path(arcname).replace("\\", "/")
                        if not _is_locked_material_reference(spec_arcname) and spec_arcname not in diff_spec_paths:
                            diff_outputs[spec_arcname] = specular_bytes
                            diff_spec_paths.add(spec_arcname)
                            record["specular"] = spec_arcname
                else:
                    hires_payloads.append((composite_name, composite_bytes))
                    existing_hires_names.add(composite_name.upper())
                    material_key = ("texture", _material_target_from_wad(composite_name))
                    record = material_records.setdefault(material_key, {})
                    if normal_bytes and "normal" not in record:
                        normal_lump = _generate_normal_lump_name(composite_name, existing_hires_names)
                        if not _is_locked_material_reference(normal_lump):
                            hires_payloads.append((normal_lump, normal_bytes))
                            record["normal"] = normal_lump
                    if specular_bytes and "specular" not in record:
                        spec_lump = _generate_specular_lump_name(composite_name, existing_hires_names)
                        if not _is_locked_material_reference(spec_lump):
                            hires_payloads.append((spec_lump, specular_bytes))
                            record["specular"] = spec_lump

        texture_def_header = "// Auto-generated by texture_upscaler scale overrides"
        texture_def_lines: List[str] = [texture_def_header, ""]
        definition_count = 0

        for patch_name in sorted(patch_replacements):
            replacement = patch_replacements[patch_name]
            if replacement.category not in {"sprite", "character", "ui"}:
                continue
            if replacement.new_width == 0 or replacement.new_height == 0:
                continue
            scale_x = replacement.scale_x
            scale_y = replacement.scale_y
            if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
                continue
            keyword = "sprite" if replacement.category in {"sprite", "character"} else "graphic"
            name_token = _format_texture_token(patch_name)
            hires_token = _format_texture_token(replacement.hires_name)
            texture_def_lines.append(
                f"{keyword} {name_token}, {replacement.new_width}, {replacement.new_height}"
            )
            texture_def_lines.append("{")
            texture_def_lines.append(f"    Patch {hires_token}, 0, 0")
            if replacement.offset_x is not None and replacement.offset_y is not None:
                texture_def_lines.append(
                    f"    Offset {int(replacement.offset_x)}, {int(replacement.offset_y)}"
                )
            texture_def_lines.append(f"    XScale {format_float(scale_x)}")
            texture_def_lines.append(f"    YScale {format_float(scale_y)}")
            texture_def_lines.append("}")
            texture_def_lines.append("")
            definition_count += 1

        composite_defs = collect_composite_textures(wad) if patch_replacements else {}
        for name in sorted(composite_defs):
            lump_name = name.upper()[:8]
            scale = composite_scales.get(lump_name)
            if not scale:
                continue
            comp = composite_defs[name]
            if not any(
                patch_replacements.get(p.name)
                and patch_replacements[p.name].category in {"sprite", "ui"}
                for p in comp.patches
            ):
                continue
            scale_x, scale_y = scale
            if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
                continue
            target_width = int(round(comp.width * scale_x)) if comp.width else 0
            target_height = int(round(comp.height * scale_y)) if comp.height else 0
            if target_width <= 0 or target_height <= 0:
                continue
            texture_def_lines.append(
                f"texture {_format_texture_token(comp.name)}, {target_width}, {target_height}"
            )
            texture_def_lines.append("{")
            texture_def_lines.append(f"    Patch {_format_texture_token(lump_name)}, 0, 0")
            texture_def_lines.append(f"    XScale {format_float(scale_x)}")
            texture_def_lines.append(f"    YScale {format_float(scale_y)}")
            texture_def_lines.append("}")
            texture_def_lines.append("")
            definition_count += 1

        if definition_count:
            texture_def_data = ("\n".join(texture_def_lines)).rstrip() + "\n"
            header_bytes = (texture_def_header + "\n").encode("ascii")
            wad.lumps = [
                lump
                for lump in wad.lumps
                if not (lump.name.upper() == "TEXTURES" and lump.data.startswith(header_bytes))
            ]
            insert_index = next(
                (i for i, lump in enumerate(wad.lumps) if lump.name.upper() == "HI_START"),
                len(wad.lumps),
            )
            wad.lumps.insert(insert_index, WadLump(name="TEXTURES", data=texture_def_data.encode("ascii")))

    if not dry_run and not diff_mode and hires_payloads:
        hi_start_idx = next(
            (i for i, lump in enumerate(wad.lumps) if lump.name.upper() == "HI_START"),
            None,
        )
        hi_end_idx = None
        if hi_start_idx is not None:
            hi_end_idx = next(
                (
                    i
                    for i in range(hi_start_idx + 1, len(wad.lumps))
                    if wad.lumps[i].name.upper() == "HI_END"
                ),
                None,
            )
            if hi_end_idx is None:
                hi_end_idx = hi_start_idx + 1
                wad.lumps.insert(hi_end_idx, WadLump(name="HI_END", data=b""))
        else:
            hi_start_idx = len(wad.lumps)
            wad.lumps.append(WadLump(name="HI_START", data=b""))
            hi_end_idx = hi_start_idx + 1
            wad.lumps.append(WadLump(name="HI_END", data=b""))

        new_names = {name.upper() for name, _ in hires_payloads}
        if hi_start_idx is not None and hi_end_idx is not None:
            i = hi_start_idx + 1
            while i < hi_end_idx:
                if wad.lumps[i].name.upper() in new_names:
                    wad.lumps.pop(i)
                    hi_end_idx -= 1
                else:
                    i += 1

        insert_idx = hi_end_idx
        for lump_name, data in hires_payloads:
            wad.lumps.insert(insert_idx, WadLump(name=lump_name, data=data))
            insert_idx += 1

    if not dry_run and not diff_mode:
        material_block = _build_material_block(material_records)
        if material_block:
            block_bytes = material_block.encode("ascii")
            inserted = False
            for lump in wad.lumps:
                if lump.name.upper() == "GLDEFS":
                    existing = lump.data or b""
                    existing = existing.rstrip(b"\r\n")
                    if existing:
                        lump.data = existing + b"\n\n" + block_bytes
                    else:
                        lump.data = block_bytes
                    inserted = True
                    break
            if not inserted:
                wad.lumps.append(WadLump(name="GLDEFS", data=block_bytes))

    hirestex_payload_diff: Optional[bytes] = None
    material_bytes_diff: Optional[bytes] = None
    if diff_mode:
        unique_diff_entries: Dict[str, Pk3ScaleEntry] = {}
        for entry in diff_entry_map.values():
            unique_diff_entries.setdefault(entry.name, entry)
        hirestex_payload_diff = build_hirestex_payload(
            unique_diff_entries.values(),
            allowed_categories=None,
            include_unscaled=True,
        )
        material_block = _build_material_block(material_records)
        material_bytes_diff = material_block.encode("ascii") if material_block else None

    if character_collector is not None and character_collector.has_characters() and character_model_config is not None:
        run_character_modeler_pipeline(
            collector=character_collector,
            config=character_model_config,
            dry_run=dry_run,
        )

    if dry_run:
        cleanup_temp_dir(temp_root, keep_temp)
        logging.info("Dry run complete. No archive was written.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if diff_mode:
        diff_files: Dict[str, bytes] = dict(diff_outputs)
        if hirestex_payload_diff:
            diff_files["HIRESTEX"] = hirestex_payload_diff
        if material_bytes_diff:
            diff_files["GLDEFS"] = material_bytes_diff
        if not diff_files:
            logging.warning("No upscaled assets were generated; writing empty diff PK3.")
        logging.info("Writing diff PK3 -> %s", dest)
        with zipfile.ZipFile(dest, "w") as output_zip:
            for arcname in sorted(diff_files.keys()):
                info = zipfile.ZipInfo(arcname)
                info.compress_type = zipfile.ZIP_DEFLATED
                output_zip.writestr(info, diff_files[arcname])
        cleanup_temp_dir(temp_root, keep_temp)
        return

    logging.info("Writing enhanced WAD -> %s", dest)
    write_wad(wad, dest)

    cleanup_temp_dir(temp_root, keep_temp)

def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        logging.error("Tkinter is required for the GUI: %s", exc)
        sys.exit(1)
    try:
        from PIL import Image, ImageTk  # type: ignore
    except ImportError as exc:
        logging.error("Pillow is required for the GUI: %s", exc)
        sys.exit(1)

    import importlib
    import queue
    import shlex
    import subprocess
    import threading

    APP_BG = "#0b0d12"
    CARD_BG = "#141923"
    ACCENT = "#43c6ac"
    TEXT_PRIMARY = "#f5f7fa"
    TEXT_SECONDARY = "#8f9bb3"
    TITLE_BG = "#141c27"
    TITLE_ACCENT = "#44d2c8"
    TITLE_TEXT = "#f5f7fa"
    DEFAULT_TEXTURE_EXTENSIONS = ".png,.jpg,.jpeg,.tga,.dds,.gif,.bmp,.webp,.tiff"
    DEFAULT_MAX_PIXELS = "300000"
    ETH_DONATION_ADDRESS = "0xB0E6226950eecE64E5E0e94B88C2282c8762808A"

    def _config_root() -> Path:
        if sys.platform.startswith("win"):
            base = os.environ.get("APPDATA")
            if base:
                return Path(base) / "DOOM Upscaler"
            return Path.home() / "AppData" / "Roaming" / "DOOM Upscaler"
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "DOOM Upscaler"
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "doom-ai-upscaler"
        return Path.home() / ".config" / "doom-ai-upscaler"

    def _config_file_path() -> Path:
        return _config_root() / "settings.ini"

    def _is_music_enabled_in_config() -> bool:
        path = _config_file_path()
        parser = configparser.ConfigParser()
        try:
            if path.is_file():
                parser.read(path, encoding="utf-8")
        except Exception:
            return True
        if parser.has_section("options"):
            try:
                return parser.getboolean("options", "music_enabled", fallback=True)
            except ValueError:
                return True
        return True

    def _ethereum_payment_uri(address: str) -> str:
        raw = address.strip()
        if not raw:
            return ""
        if raw.lower().startswith("ethereum:"):
            return raw
        return f"ethereum:{raw}"

    def _generate_eth_qr_image(address: str) -> Optional[Image.Image]:
        payload = _ethereum_payment_uri(address)
        if not payload:
            return None

        pil_image: Optional[Image.Image] = None

        if qrcode is not None:
            try:
                qr = qrcode.QRCode(
                    version=None,
                    error_correction=qrcode.constants.ERROR_CORRECT_Q,
                    box_size=10,
                    border=4,
                )
                qr.add_data(payload)
                qr.make(fit=True)
                image = qr.make_image(fill_color="black", back_color="white")
                if hasattr(image, "get_image"):
                    pil_image = image.get_image().convert("RGB")  # type: ignore[attr-defined]
                elif hasattr(image, "convert"):
                    pil_image = image.convert("RGB")  # type: ignore[assignment]
                else:
                    pil_image = Image.new("RGB", (256, 256), "white")
            except Exception as exc:  # noqa: BLE001
                logging.debug("Unable to generate ETH QR code via qrcode module: %s", exc)
                pil_image = None

        if pil_image is None:
            if _QrCodeGenClass is None or _QrSegmentGenClass is None:
                _ensure_embedded_qrcodegen()
            qr_cls = _QrCodeGenClass
            if qr_cls is not None:
                try:
                    qr_obj = qr_cls.encode_text(payload, qr_cls.Ecc.QUARTILE)
                except Exception as exc:  # noqa: BLE001
                    logging.debug("Embedded QR generator failed: %s", exc)
                    qr_obj = None
                if qr_obj is not None:
                    module_size = qr_obj.get_size()
                    quiet_zone = 4
                    scale = 4
                    size = (module_size + quiet_zone * 2) * scale
                    fallback_image = Image.new("RGB", (size, size), "white")
                    pixels = fallback_image.load()
                    if pixels is None:
                        logging.debug("Unable to access QR fallback pixel buffer.")
                    else:
                        for y in range(module_size):
                            for x in range(module_size):
                                if qr_obj.get_module(x, y):
                                    start_x = (x + quiet_zone) * scale
                                    start_y = (y + quiet_zone) * scale
                                    for dy in range(scale):
                                        for dx in range(scale):
                                            pixels[start_x + dx, start_y + dy] = (0, 0, 0)
                        pil_image = fallback_image
        if pil_image is None:
            logging.debug("QR code generation skipped: no available generator.")
            return None

        width, height = pil_image.size
        max_dimension = 128
        if width > max_dimension or height > max_dimension:
            resampling = getattr(Image, "Resampling", Image)
            lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
            scale = min(max_dimension / max(width, 1), max_dimension / max(height, 1))
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            try:
                pil_image = pil_image.resize(new_size, lanczos)
            except Exception as exc:  # noqa: BLE001
                logging.debug("Unable to resize QR code image: %s", exc)
        return pil_image

    def _expand_asset_roots(*roots: Path) -> Tuple[Path, ...]:
        expanded: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            for candidate in (root, root / "assets"):
                try:
                    resolved = candidate.resolve()
                except OSError:
                    resolved = candidate
                if resolved not in seen:
                    seen.add(resolved)
                    expanded.append(resolved)
        return tuple(expanded)

    def _show_splash_window(exec_dir: Path, resource_dir: Path) -> None:
        splash_roots = _expand_asset_roots(resource_dir, exec_dir, Path.cwd())
        splash_candidates = [
            root / "splashscreen.png"
            for root in splash_roots
        ] + [
            root / "DOOM Upscaler.png"
            for root in splash_roots
        ]
        splash_path = next((path for path in splash_candidates if path.is_file()), None)
        if splash_path is None:
            return
        try:
            splash_image = Image.open(splash_path)
        except Exception as exc:
            logging.debug("Failed to open splash image %s: %s", splash_path, exc)
            return

        resampling = getattr(Image, "Resampling", Image)
        lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
        max_width = 800
        max_height = 450
        width, height = splash_image.size
        scale_factor = 1.0
        if width > max_width or height > max_height:
            scale_factor = min(max_width / width, max_height / height)
        if scale_factor != 1.0:
            target_size = (int(width * scale_factor), int(height * scale_factor))
            try:
                splash_image = splash_image.resize(target_size, lanczos)
                width, height = splash_image.size
            except Exception:
                pass

        splash_root = tk.Tk()
        splash_root.overrideredirect(True)
        try:
            splash_root.attributes("-topmost", True)
        except Exception:
            pass

        photo = ImageTk.PhotoImage(splash_image)
        width, height = splash_image.size
        screen_width = splash_root.winfo_screenwidth()
        screen_height = splash_root.winfo_screenheight()
        x_pos = int((screen_width - width) / 2)
        y_pos = int((screen_height - height) / 2)
        splash_root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

        label = tk.Label(splash_root, image=photo, borderwidth=0, highlightthickness=0)
        label.image = photo
        label.pack()

        splash_root.after(2500, splash_root.destroy)
        splash_root.mainloop()

    def _runtime_directories() -> Tuple[Path, Path]:
        if getattr(sys, "frozen", False):
            exec_dir = Path(sys.executable).resolve().parent
            resource_dir = Path(getattr(sys, "_MEIPASS", exec_dir))
        else:
            exec_dir = Path(__file__).resolve().parent
            resource_dir = exec_dir
        return exec_dir, resource_dir

    class PixelShadowsPlayer:
        """Loop the bundled Pixel Shadows track via pygame, with a WinMM fallback on Windows."""

        def __init__(self, search_roots: Tuple[Path, ...]) -> None:
            self._search_roots = _expand_asset_roots(*search_roots)
            self._music_path = self._resolve_track()
            self._thread: Optional[threading.Thread] = None
            self._stop_event = threading.Event()
            self._pygame = None
            self._backend: Optional[str] = None
            self._winmm_alias: Optional[str] = None
            self._winmm_sendstring = None
            self.error: Optional[str] = None

        def _resolve_track(self) -> Optional[Path]:
            for root in self._search_roots:
                candidate = root / "Pixel Shadows.mp3"
                if candidate.is_file():
                    return candidate
            self.error = "Missing soundtrack: Pixel Shadows.mp3"
            return None

        def start(self) -> None:
            if self._music_path is None or self._thread is not None:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="pixel-shadows", daemon=True)
            self._thread.start()

        def stop(self) -> None:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None
            self._shutdown_backend()

        def is_running(self) -> bool:
            return self._thread is not None and self._thread.is_alive()

        def _loop(self) -> None:
            try:
                self._ensure_backend()
                if self.error:
                    return
                if self._backend == "pygame":
                    self._loop_with_pygame()
                elif self._backend == "winmm":
                    self._loop_with_winmm()
            except Exception as exc:
                self.error = f"Unable to play Pixel Shadows.mp3: {exc}"
            finally:
                self._thread = None

        def _loop_with_pygame(self) -> None:
            assert self._pygame is not None
            assert self._music_path is not None
            mixer = self._pygame.mixer
            mixer.music.load(str(self._music_path))
            mixer.music.play(-1)
            logging.debug("Pixel Shadows soundtrack playing via pygame mixer.")
            while not self._stop_event.wait(0.5):
                if not mixer.music.get_busy():
                    mixer.music.play(-1)

        def _loop_with_winmm(self) -> None:
            assert self._music_path is not None
            alias = self._winmm_alias or f"ps_{os.getpid()}_{int(time.time() * 1000)}"
            self._winmm_alias = alias
            path = str(self._music_path).replace('"', '""')
            try:
                self._winmm_send(f'open "{path}" type mpegvideo alias {alias}')
                self._winmm_send(f"play {alias} repeat")
                logging.debug("Pixel Shadows soundtrack playing via WinMM fallback.")
                while not self._stop_event.wait(0.5):
                    pass
            finally:
                self._winmm_send(f"stop {alias}", ignore_errors=True)
                self._winmm_send(f"close {alias}", ignore_errors=True)

        def _ensure_backend(self) -> None:
            if self.error:
                return
            if self._backend == "pygame" or self._backend == "winmm":
                return
            pygame_module = None
            try:
                pygame_module = importlib.import_module("pygame")
            except ImportError:
                pygame_module = None
            except Exception as exc:  # noqa: BLE001 - optional dependency diagnostics
                logging.debug("Unable to import pygame: %s", exc)
            if pygame_module is not None:
                self._pygame = pygame_module
                try:
                    if not self._pygame.mixer.get_init():
                        self._pygame.mixer.init()
                except Exception as exc:
                    self.error = f"Unable to initialize pygame mixer: {exc}"
                    self._pygame = None
                    return
                self._backend = "pygame"
                self.error = None
                return
            if os.name == "nt":
                try:
                    self._setup_winmm_backend()
                    self._backend = "winmm"
                    self.error = None
                    return
                except Exception as exc:
                    logging.debug("Unable to initialize WinMM soundtrack fallback: %s", exc)
            self.error = "Install pygame to enable the Pixel Shadows soundtrack (pip install pygame)."

        def _setup_winmm_backend(self) -> None:
            if os.name != "nt":
                raise RuntimeError("WinMM fallback is only available on Windows.")
            if self._music_path is None:
                raise RuntimeError("Missing soundtrack: Pixel Shadows.mp3")
            try:
                self._winmm_sendstring = ctypes.windll.winmm.mciSendStringW  # type: ignore[attr-defined]
            except AttributeError as exc:
                raise RuntimeError("Windows multimedia APIs are unavailable.") from exc

        def _shutdown_backend(self) -> None:
            if self._backend == "pygame" and self._pygame is not None:
                try:
                    if self._pygame.mixer.get_init():
                        self._pygame.mixer.music.stop()
                        self._pygame.mixer.quit()
                except Exception:
                    pass
                self._pygame = None
            elif self._backend == "winmm":
                if self._winmm_alias:
                    self._winmm_send(f"stop {self._winmm_alias}", ignore_errors=True)
                    self._winmm_send(f"close {self._winmm_alias}", ignore_errors=True)
                self._winmm_alias = None
                self._winmm_sendstring = None
            self._backend = None

        def _winmm_send(self, command: str, *, ignore_errors: bool = False) -> None:
            if self._winmm_sendstring is None:
                if ignore_errors:
                    return
                raise RuntimeError("WinMM backend has not been initialized.")
            result = self._winmm_sendstring(command, None, 0, None)  # type: ignore[call-arg]
            if result != 0 and not ignore_errors:
                raise RuntimeError(f"WinMM command failed ({result}): {command}")
            if result != 0 and ignore_errors:
                logging.debug("WinMM command failed (%s): %s", result, command)

    class UpscaleWorker(threading.Thread):
        """Runs the CLI in the background and streams stdout lines into a queue."""

        def __init__(self, command: list[str], output_queue: "queue.Queue[str]") -> None:
            super().__init__(daemon=True)
            self._command = command
            self._output_queue = output_queue
            self._process: Optional[subprocess.Popen[str]] = None
            self._abort_requested = threading.Event()

        def _force_stop(self, process: subprocess.Popen[str]) -> None:
            if process.poll() is not None:
                return
            if os.name == "nt":
                ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                if ctrl_break is not None:
                    try:
                        process.send_signal(ctrl_break)
                    except Exception:
                        pass
                    time.sleep(0.1)
                if process.poll() is None:
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    except Exception:
                        pass
            else:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except Exception:
                    try:
                        process.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                time.sleep(0.1)
            if process.poll() is not None:
                return
            try:
                process.terminate()
            except Exception:
                pass
            time.sleep(0.1)
            if process.poll() is None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except Exception:
                    pass
                time.sleep(0.1)
            if process.poll() is not None:
                return
            try:
                process.kill()
            except Exception:
                pass
            if process.poll() is None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass

        def run(self) -> None:
            try:
                creationflags = 0
                if os.name == "nt":
                    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                preexec_fn = None
                if os.name != "nt":
                    preexec_fn = os.setsid
                process = subprocess.Popen(
                    self._command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                )
            except Exception as exc:  # noqa: BLE001
                self._output_queue.put(f"[launcher] Failed to start upscaler: {exc}")
                self._output_queue.put("__RETURN_CODE__1")
                return

            self._process = process

            if self._abort_requested.is_set() and process.poll() is None:
                self._force_stop(process)

            assert process.stdout is not None
            for line in process.stdout:
                self._output_queue.put(line.rstrip("\n"))
            return_code = process.wait()
            self._process = None
            if self._abort_requested.is_set():
                self._output_queue.put("__ABORTED__")
            self._output_queue.put(f"__RETURN_CODE__{return_code}")

        def abort(self) -> None:
            self._abort_requested.set()
            process = self._process
            if process is None:
                return
            self._force_stop(process)

    class TextureUpscalerLauncher(tk.Tk):
        def __init__(self, music_player: Optional["PixelShadowsPlayer"] = None) -> None:
            super().__init__()
            self.title("DOOM Upscaler")
            self.configure(bg=APP_BG)
            self.resizable(False, False)
            self.overrideredirect(True)
            self._is_dragging = False
            self._drag_offset_x = 0
            self._drag_offset_y = 0
            self._exec_dir, self._resource_dir = _runtime_directories()

            self._file_var = tk.StringVar()
            self._scale_var = tk.DoubleVar(value=4.0)
            self._scale_display_var = tk.StringVar(value="4x")
            self._status_var = tk.StringVar(value="Ready to enhance your textures.")

            self._output_dir_var = tk.StringVar()
            self._output_mode_var = tk.StringVar(value="diff")
            self._enable_sprite_materials_var = tk.BooleanVar(value=False)
            self._disable_materials_var = tk.BooleanVar(value=False)
            self._enable_post_sharpen_var = tk.BooleanVar(value=True)
            self._keep_temp_var = tk.BooleanVar(value=False)
            self._dry_run_var = tk.BooleanVar(value=False)
            self._verbose_var = tk.BooleanVar(value=False)
            self._music_enabled_var = tk.BooleanVar(value=True)

            self._realesrgan_bin_var = tk.StringVar()
            self._realesrgan_pth_var = tk.StringVar(value=DEFAULT_PTH_MODEL_NAME)
            self._realesrgan_gpu_var = tk.StringVar(value="auto")
            self._realesrgan_tile_var = tk.StringVar(value="0")
            self._realesrgan_tile_pad_var = tk.StringVar(value="10")
            self._gpu_options = self._detect_gpu_choices()
            if self._gpu_options:
                self._realesrgan_gpu_var.set(self._gpu_options[0])
            else:
                self._gpu_options = ["auto"]
                self._realesrgan_gpu_var.set("auto")

            self._detail_scale_var = tk.StringVar()
            self._esrgan_strength_var = tk.StringVar(value="0.5")
            self._esrgan_detail_limit_var = tk.StringVar(value="0.0")
            self._max_pixels_var = tk.StringVar(value=DEFAULT_MAX_PIXELS)
            self._texture_exts_var = tk.StringVar(value=DEFAULT_TEXTURE_EXTENSIONS)
            self._character_keywords_var = tk.StringVar()
            self._skip_categories_var = tk.StringVar()
            self._skip_categories_display = tk.StringVar(value="(none)")
            self._skip_category_vars: Dict[str, tk.BooleanVar] = {
                category: tk.BooleanVar(value=False) for category in TEXTURE_CATEGORY_CHOICES
            }
            self._skip_categories_menu: Optional[tk.Menu] = None

            self._config_path = _config_file_path()
            self._load_config_from_disk()

            self._log_queue: "queue.Queue[str]" = queue.Queue()
            self._worker: Optional[UpscaleWorker] = None
            self._abort_requested = False
            self._initial_focus_applied = False

            self._content_frame: Optional[ttk.Frame] = None
            self._run_button: Optional[ttk.Button] = None
            self._log_box: Optional[tk.Text] = None
            self._log_scrollbar: Optional[ttk.Scrollbar] = None
            self._log_toggle_button: Optional[ttk.Button] = None
            self._log_toggle_container: Optional[ttk.Frame] = None
            self._log_frame: Optional[ttk.Frame] = None
            self._log_visible = tk.BooleanVar(value=False)
            self._arcade_mode = tk.BooleanVar(value=False)
            self._arcade_container: Optional[ttk.Frame] = None
            self._arcade_panel: Optional["UpscalerApp._ArcadePanel"] = None
            self._arcade_button: Optional[ttk.Button] = None
            self._music_toggle: Optional[ttk.Checkbutton] = None
            self._sprite_materials_check: Optional[ttk.Checkbutton] = None
            self._disable_materials_check: Optional[ttk.Checkbutton] = None
            self._gpu_combo: Optional[ttk.Combobox] = None
            self._file_entry: Optional[ttk.Entry] = None
            self._pth_combo: Optional[ttk.Combobox] = None
            self._qr_photo: Optional[ImageTk.PhotoImage] = None
            self._icon_photo: Optional[tk.PhotoImage] = None
            self._collapsed_width: Optional[int] = None
            self._expanded_width: Optional[int] = None
            self._config_controls: List[tk.Widget] = []
            self._config_control_states: Dict[tk.Widget, str] = {}
            self._progress_processed = 0
            self._progress_total = 0

            self._music_search_roots: Tuple[Path, ...] = (
                self._resource_dir,
                self._exec_dir,
                Path.cwd(),
            )
            self.music_player: Optional[PixelShadowsPlayer] = music_player
            self._music_error_reported = False
            self._music_has_played = False
            if self._music_enabled_var.get():
                try:
                    player = self._ensure_music_player()
                    if player is not None:
                        player.start()
                except Exception as exc:
                    logging.debug("Unable to start soundtrack: %s", exc)
            else:
                if self.music_player is not None:
                    self.music_player.stop()

            self._set_window_icon()
            self._create_title_bar()
            self._build_ui()
            self._setup_drag_and_drop()
            self._center_on_screen()
            self._poll_log_queue()
            self._poll_music_status()
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self.bind("<Map>", self._on_map)
            self.bind("<Unmap>", self._on_unmap)
            self.after(50, lambda: self.overrideredirect(True))
            self.after(120, self._focus_main_window)

        def _load_config_from_disk(self) -> None:
            path = getattr(self, "_config_path", None)
            if not isinstance(path, Path) or not path.is_file():
                # Ensure display text matches current scale defaults.
                self._scale_display_var.set(f"{int(round(self._scale_var.get()))}x")
                self._update_skip_categories_selection()
                return

            parser = configparser.ConfigParser()
            try:
                parser.read(path, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - malformatted config is non-fatal
                logging.debug("Unable to read config file %s: %s", path, exc)
                self._scale_display_var.set(f"{int(round(self._scale_var.get()))}x")
                self._update_skip_categories_selection()
                return

            if parser.has_section("general"):
                general = parser["general"]

                value = general.get("file_path")
                if value is not None:
                    self._file_var.set(value.strip())

                value = general.get("output_dir")
                if value is not None:
                    self._output_dir_var.set(value.strip())

                value = general.get("output_mode")
                if value:
                    self._output_mode_var.set(value.strip())

                value = general.get("scale")
                if value:
                    try:
                        numeric = float(value)
                        self._scale_var.set(numeric)
                        self._scale_display_var.set(f"{int(round(numeric))}x")
                    except ValueError:
                        logging.debug("Invalid scale value in config: %s", value)

                value = general.get("detail_scale")
                if value is not None:
                    self._detail_scale_var.set(value.strip())

                value = general.get("esrgan_strength")
                if value is not None:
                    self._esrgan_strength_var.set(value.strip())

                value = general.get("esrgan_detail_limit")
                if value is not None:
                    self._esrgan_detail_limit_var.set(value.strip())

                value = general.get("max_pixels")
                if value is not None:
                    self._max_pixels_var.set(value.strip())

                value = general.get("texture_extensions")
                if value is not None:
                    self._texture_exts_var.set(value.strip())

                value = general.get("character_keywords")
                if value is not None:
                    self._character_keywords_var.set(value.strip())

            if parser.has_section("options"):
                options = parser["options"]
                bool_mappings: Sequence[Tuple[tk.BooleanVar, str]] = (
                    (self._enable_sprite_materials_var, "enable_sprite_materials"),
                    (self._disable_materials_var, "disable_materials"),
                    (self._enable_post_sharpen_var, "enable_post_sharpen"),
                    (self._keep_temp_var, "keep_temp"),
                    (self._dry_run_var, "dry_run"),
                    (self._verbose_var, "verbose"),
                    (self._music_enabled_var, "music_enabled"),
                )
                for target, key in bool_mappings:
                    try:
                        target.set(options.getboolean(key, fallback=target.get()))
                    except ValueError:
                        logging.debug("Invalid boolean for %s in config; keeping default.", key)

            if parser.has_section("realesrgan"):
                realesrgan = parser["realesrgan"]

                value = realesrgan.get("binary_path")
                if value is not None:
                    self._realesrgan_bin_var.set(value.strip())

                value = realesrgan.get("pth_path")
                if value is not None:
                    self._realesrgan_pth_var.set(value.strip())

                value = realesrgan.get("gpu_index")
                if value:
                    clean = value.strip()
                    if clean and clean not in self._gpu_options:
                        self._gpu_options.append(clean)
                    if clean:
                        self._realesrgan_gpu_var.set(clean)

                value = realesrgan.get("tile_size")
                if value is not None:
                    self._realesrgan_tile_var.set(value.strip())

                value = realesrgan.get("tile_padding")
                if value is not None:
                    self._realesrgan_tile_pad_var.set(value.strip())

            if parser.has_section("skip_categories"):
                skip_section = parser["skip_categories"]
                selected_raw = skip_section.get("selected", "")
                selected = {
                    name.strip() for name in selected_raw.split(",") if name.strip() in self._skip_category_vars
                }
                for name, var in self._skip_category_vars.items():
                    var.set(name in selected)

            # Final pass to ensure derived state aligns with loaded values.
            self._scale_display_var.set(f"{int(round(self._scale_var.get()))}x")
            self._update_skip_categories_selection()

        def _save_config_to_disk(self) -> None:
            path = getattr(self, "_config_path", None)
            if not isinstance(path, Path):
                return

            parser = configparser.ConfigParser()

            self._update_skip_categories_selection()

            parser["general"] = {
                "file_path": self._file_var.get().strip(),
                "output_dir": self._output_dir_var.get().strip(),
                "output_mode": self._output_mode_var.get().strip(),
                "scale": str(self._scale_var.get()),
                "detail_scale": self._detail_scale_var.get().strip(),
                "esrgan_strength": self._esrgan_strength_var.get().strip(),
                "esrgan_detail_limit": self._esrgan_detail_limit_var.get().strip(),
                "max_pixels": self._max_pixels_var.get().strip(),
                "texture_extensions": self._texture_exts_var.get().strip(),
                "character_keywords": self._character_keywords_var.get().strip(),
            }

            parser["options"] = {
                "enable_sprite_materials": str(self._enable_sprite_materials_var.get()),
                "disable_materials": str(self._disable_materials_var.get()),
                "enable_post_sharpen": str(self._enable_post_sharpen_var.get()),
                "keep_temp": str(self._keep_temp_var.get()),
                "dry_run": str(self._dry_run_var.get()),
                "verbose": str(self._verbose_var.get()),
                "music_enabled": str(self._music_enabled_var.get()),
            }

            parser["realesrgan"] = {
                "binary_path": self._realesrgan_bin_var.get().strip(),
                "pth_path": self._realesrgan_pth_var.get().strip(),
                "gpu_index": self._realesrgan_gpu_var.get().strip(),
                "tile_size": self._realesrgan_tile_var.get().strip(),
                "tile_padding": self._realesrgan_tile_pad_var.get().strip(),
            }

            parser["skip_categories"] = {
                "selected": self._skip_categories_var.get().strip(),
            }

            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    parser.write(handle)
            except Exception as exc:  # noqa: BLE001 - config persistence shouldn't crash the app
                logging.debug("Unable to write config file %s: %s", path, exc)

        def _center_on_screen(self) -> None:
            self.update_idletasks()
            width = self.winfo_width()
            height = self.winfo_height()
            if width <= 0 or height <= 0:
                return
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x_pos = max(int((screen_width - width) / 2), 0)
            y_pos = max(int((screen_height - height) / 2), 0)
            self.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

        def _create_title_bar(self) -> None:
            self._title_bar = tk.Frame(self, bg=TITLE_BG, relief="flat", height=38, bd=0, highlightthickness=0)
            self._title_bar.pack(fill="x", side="top")

            title_label = tk.Label(
                self._title_bar,
                text="DOOM Upscaler",
                fg=TITLE_TEXT,
                bg=TITLE_BG,
                font=("Segoe UI", 11, "bold"),
                padx=12,
                pady=6,
            )
            title_label.pack(side="left")

            button_container = tk.Frame(self._title_bar, bg=TITLE_BG, bd=0, highlightthickness=0)
            button_container.pack(side="right", padx=(0, 6))

            title_button_font = ("Segoe UI Symbol", 13)
            close_button = tk.Label(
                button_container,
                text="\u2715",
                fg=TITLE_TEXT,
                bg=TITLE_BG,
                font=title_button_font,
                padx=10,
                pady=2,
            )
            close_button.pack(side="right", padx=(0, 2))
            close_button.bind("<Button-1>", lambda _: self._on_close())
            close_button.bind("<Enter>", lambda e: e.widget.config(bg="#d64550", fg="#ffffff"))
            close_button.bind("<Leave>", lambda e: e.widget.config(bg=TITLE_BG, fg=TITLE_TEXT))

            minimize_button = tk.Label(
                button_container,
                text="\u2212",
                fg=TITLE_TEXT,
                bg=TITLE_BG,
                font=title_button_font,
                padx=10,
                pady=2,
            )
            minimize_button.pack(side="right", padx=2)
            minimize_button.bind("<Button-1>", lambda _: self._minimize_window())
            minimize_button.bind("<Enter>", lambda e: e.widget.config(bg="#1e2633"))
            minimize_button.bind("<Leave>", lambda e: e.widget.config(bg=TITLE_BG))

            for widget in (self._title_bar, title_label):
                widget.bind("<ButtonPress-1>", self._start_window_move)
                widget.bind("<B1-Motion>", self._on_window_move)
                widget.bind("<ButtonRelease-1>", self._stop_window_move)

        def _build_ui(self) -> None:
            style = ttk.Style(self)
            style.theme_use("clam")
            style.configure("TFrame", background=APP_BG)
            style.configure("Card.TFrame", background=CARD_BG)
            style.configure("TLabel", background=APP_BG, foreground=TEXT_PRIMARY)
            style.configure("Secondary.TLabel", background=APP_BG, foreground=TEXT_SECONDARY)
            style.configure("CardLabel.TLabel", background=CARD_BG, foreground=TEXT_PRIMARY)
            style.layout("Upscaler.TNotebook", [("Notebook.client", {"sticky": "nsew"})])
            style.configure(
                "Upscaler.TNotebook",
                background=APP_BG,
                borderwidth=0,
                padding=0,
                tabmargins=(0, 0, 0, 0),
            )
            style.configure(
                "Upscaler.TNotebook.Tab",
                background="#1a2331",
                foreground=TEXT_PRIMARY,
                padding=(16, 6),
                font=("Segoe UI", 10, "bold"),
            )
            style.map(
                "Upscaler.TNotebook.Tab",
                background=[("selected", TITLE_ACCENT), ("active", "#223041")],
                foreground=[("selected", "#091015"), ("disabled", TEXT_SECONDARY)],
                padding=[
                    ("selected", (16, 10)),
                    ("active", (16, 8)),
                    ("!selected", (16, 6)),
                ],
            )
            style.configure("Accent.TButton", background=ACCENT, foreground="#101218")
            style.map(
                "Accent.TButton",
                background=[("active", ACCENT), ("disabled", "#2b3142")],
                foreground=[("disabled", "#4c566a")],
            )
            style.configure(
                "ArcadeLaunch.TButton",
                background=TITLE_ACCENT,
                foreground="#07131c",
                font=("Segoe UI", 10, "bold"),
                padding=(18, 6),
                borderwidth=0,
                relief="flat",
            )
            style.map(
                "ArcadeLaunch.TButton",
                background=[("pressed", "#2ea399"), ("active", "#5ff0e6"), ("disabled", "#1f2b36")],
                foreground=[("disabled", "#4c566a")],
            )
            style.configure("Card.TButton", background="#1f2733", foreground=TEXT_PRIMARY)
            style.map(
                "Card.TButton",
                background=[("active", "#272f3d"), ("disabled", "#1a202b")],
                foreground=[("disabled", "#4c566a")],
            )
            style.configure(
                "TerminalToggle.TButton",
                background=APP_BG,
                foreground=TEXT_PRIMARY,
                borderwidth=0,
                relief="flat",
                padding=0,
            )
            style.map(
                "TerminalToggle.TButton",
                background=[("active", APP_BG), ("disabled", APP_BG)],
                foreground=[("disabled", TEXT_SECONDARY)],
            )
            style.configure("Card.TCheckbutton", background=CARD_BG, foreground=TEXT_PRIMARY)
            style.map(
                "Card.TCheckbutton",
                background=[("active", CARD_BG), ("disabled", CARD_BG)],
                foreground=[("disabled", TEXT_SECONDARY)],
            )
            style.configure("Card.TRadiobutton", background=CARD_BG, foreground=TEXT_PRIMARY)
            style.map(
                "Card.TRadiobutton",
                background=[("active", CARD_BG), ("disabled", CARD_BG)],
                foreground=[("disabled", TEXT_SECONDARY)],
            )
            style.configure("Card.TMenubutton", background=CARD_BG, foreground=TEXT_PRIMARY, relief="flat")
            style.map(
                "Card.TMenubutton",
                background=[("active", "#1c2431"), ("disabled", CARD_BG)],
                foreground=[("disabled", TEXT_SECONDARY)],
            )
            style.configure("Card.TLabelframe", background=CARD_BG, foreground=TEXT_PRIMARY)
            style.configure("Card.TLabelframe.Label", background=CARD_BG, foreground=TEXT_PRIMARY)

            content = ttk.Frame(self, style="TFrame", padding=0)
            content.pack(fill="both", expand=True)
            content.columnconfigure(0, weight=3)
            content.columnconfigure(1, weight=2)
            content.columnconfigure(2, weight=0)
            content.rowconfigure(0, weight=1)
            self._content_frame = content

            notebook_container = ttk.Frame(content, padding=(12, 12, 8, 12), style="TFrame")
            notebook_container.grid(row=0, column=0, sticky="nsew")
            notebook_container.columnconfigure(0, weight=1)
            notebook_container.rowconfigure(0, weight=1)

            notebook = ttk.Notebook(notebook_container, style="Upscaler.TNotebook")
            notebook.grid(row=0, column=0, sticky="nsew")

            general_tab = ttk.Frame(notebook, style="TFrame")
            advanced_tab = ttk.Frame(notebook, style="TFrame")
            about_tab = ttk.Frame(notebook, style="TFrame")
            notebook.add(general_tab, text="General")
            notebook.add(advanced_tab, text="Advanced")
            notebook.add(about_tab, text="About")

            general_tab.columnconfigure(0, weight=1)
            general_tab.rowconfigure(0, weight=1)
            advanced_tab.columnconfigure(0, weight=1)
            advanced_tab.rowconfigure(0, weight=1)
            about_tab.columnconfigure(0, weight=1)
            about_tab.rowconfigure(0, weight=1)

            card = ttk.Frame(general_tab, padding=12, style="Card.TFrame")
            card.grid(row=0, column=0, sticky="nsew")
            card.columnconfigure(1, weight=1)
            card.rowconfigure(6, weight=0)
            card.rowconfigure(7, weight=1)

            ttk.Label(card, text="WAD/PK3 file", style="CardLabel.TLabel").grid(row=0, column=0, sticky="w")
            self._file_entry = ttk.Entry(card, textvariable=self._file_var, width=30)
            self._file_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 12))

            browse = ttk.Button(card, text="Browse...", style="Card.TButton", command=self._select_file)
            browse.grid(row=1, column=2, padx=(8, 0), pady=(6, 12))

            ttk.Label(card, text="Output directory", style="CardLabel.TLabel").grid(row=2, column=0, sticky="w")
            out_entry = ttk.Entry(card, textvariable=self._output_dir_var, width=30)
            out_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 12))
            ttk.Button(card, text="Browse...", style="Card.TButton", command=self._browse_output_dir).grid(
                row=3, column=2, padx=(8, 0), pady=(6, 12)
            )

            ttk.Label(card, text="Final scale", style="CardLabel.TLabel").grid(row=4, column=0, sticky="w")

            scale_slider = ttk.Scale(
                card,
                from_=1,
                to=8,
                orient="horizontal",
                variable=self._scale_var,
                command=self._on_scale_change,
            )
            scale_slider.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))

            ttk.Label(card, textvariable=self._scale_display_var, style="CardLabel.TLabel").grid(
                row=5,
                column=2,
                padx=(8, 0),
            )

            mode_frame = ttk.Frame(card, style="Card.TFrame")
            mode_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 0))
            mode_frame.columnconfigure(0, weight=1)
            mode_frame.columnconfigure(1, weight=1)
            ttk.Label(mode_frame, text="Output mode", style="CardLabel.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
            ttk.Radiobutton(
                mode_frame,
                text="Diff (PK3 with changes only)",
                value="diff",
                variable=self._output_mode_var,
                style="Card.TRadiobutton",
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))
            ttk.Radiobutton(
                mode_frame,
                text="Full (rewrite archive)",
                value="full",
                variable=self._output_mode_var,
                style="Card.TRadiobutton",
            ).grid(row=1, column=1, sticky="w", padx=(12, 0), pady=(4, 0))

            options_frame = ttk.Frame(card, style="Card.TFrame")
            options_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
            options_frame.columnconfigure(0, weight=1)
            sprite_materials_cb = ttk.Checkbutton(
                options_frame,
                text="Generate sprite/UI materials",
                variable=self._enable_sprite_materials_var,
                style="Card.TCheckbutton",
            )
            sprite_materials_cb.grid(row=0, column=0, sticky="w")
            self._sprite_materials_check = sprite_materials_cb
            disable_materials_cb = ttk.Checkbutton(
                options_frame,
                text="Disable material generation entirely",
                variable=self._disable_materials_var,
                command=self._on_disable_materials_toggle,
                style="Card.TCheckbutton",
            )
            disable_materials_cb.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self._disable_materials_check = disable_materials_cb
            ttk.Checkbutton(
                options_frame,
                text="Keep temporary workspace",
                variable=self._keep_temp_var,
                style="Card.TCheckbutton",
            ).grid(row=2, column=0, sticky="w")
            ttk.Checkbutton(
                options_frame,
                text="Dry run (no files written)",
                variable=self._dry_run_var,
                style="Card.TCheckbutton",
            ).grid(row=3, column=0, sticky="w")
            ttk.Checkbutton(
                options_frame,
                text="Verbose logging",
                variable=self._verbose_var,
                style="Card.TCheckbutton",
            ).grid(row=4, column=0, sticky="w")
            ttk.Checkbutton(
                options_frame,
                text="Apply post-sharpen pass",
                variable=self._enable_post_sharpen_var,
                style="Card.TCheckbutton",
            ).grid(row=5, column=0, sticky="w")
            music_toggle = ttk.Checkbutton(
                options_frame,
                text="Enable background music",
                variable=self._music_enabled_var,
                command=self._on_music_toggle,
                style="Card.TCheckbutton",
            )
            music_toggle._allow_during_run = True  # type: ignore[attr-defined]
            music_toggle.grid(row=6, column=0, sticky="w")
            self._music_toggle = music_toggle
            self._on_disable_materials_toggle()

            status_label = ttk.Label(
                card,
                textvariable=self._status_var,
                style="Secondary.TLabel",
                wraplength=300,
                justify="left",
            )
            status_label.grid(row=8, column=0, columnspan=3, sticky="w", pady=(14, 6))

            run_button = ttk.Button(
                card,
                text="Enhance Textures",
                style="Accent.TButton",
                command=self._launch_upscaler,
            )
            run_button._allow_during_run = True  # type: ignore[attr-defined]
            run_button.grid(row=9, column=0, columnspan=3, sticky="ew")
            self._run_button = run_button

            advanced_card = ttk.Frame(advanced_tab, padding=12, style="Card.TFrame")
            advanced_card.grid(row=0, column=0, sticky="nsew")
            advanced_card.columnconfigure(1, weight=1)

            ttk.Label(advanced_card, text="Detail scale (leave blank to auto)", style="CardLabel.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Spinbox(
                advanced_card,
                from_=1,
                to=16,
                increment=1,
                textvariable=self._detail_scale_var,
                width=9,
            ).grid(row=0, column=1, sticky="w", pady=(6, 10))

            realesrgan_frame = ttk.LabelFrame(
                advanced_card,
                text="RealESRGAN backend",
                style="Card.TLabelframe",
                padding=8,
            )
            realesrgan_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 12))
            realesrgan_frame.columnconfigure(1, weight=1)

            ttk.Label(realesrgan_frame, text="RealESRGAN .pth model", style="CardLabel.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            self._pth_combo = ttk.Combobox(
                realesrgan_frame,
                textvariable=self._realesrgan_pth_var,
                width=26,
                values=["./models/BSRGAN.pth", "./models/4x-UltraSharp.pth"],
            )
            self._pth_combo.grid(row=0, column=1, sticky="ew", pady=(6, 6))
            ttk.Button(
                realesrgan_frame,
                text="Browse...",
                style="Card.TButton",
                command=self._browse_realesrgan_pth,
            ).grid(row=0, column=2, padx=(8, 0), pady=(6, 6))

            ttk.Label(realesrgan_frame, text="realesrgan-ncnn-vulkan executable", style="CardLabel.TLabel").grid(
                row=1, column=0, sticky="w"
            )
            ttk.Combobox(
                realesrgan_frame,
                textvariable=self._realesrgan_bin_var,
                width=26,
                values=["realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"],
            ).grid(row=1, column=1, sticky="ew", pady=(6, 6))
            ttk.Button(
                realesrgan_frame,
                text="Browse...",
                style="Card.TButton",
                command=self._browse_realesrgan_bin,
            ).grid(row=1, column=2, padx=(8, 0), pady=(6, 6))

            ttk.Label(realesrgan_frame, text="GPU index", style="CardLabel.TLabel").grid(row=2, column=0, sticky="w")
            gpu_combo = ttk.Combobox(
                realesrgan_frame,
                textvariable=self._realesrgan_gpu_var,
                width=8,
                values=self._gpu_options,
                state="readonly",
            )
            gpu_combo.grid(row=2, column=1, sticky="w", pady=(6, 6))
            self._gpu_combo = gpu_combo

            ttk.Label(realesrgan_frame, text="Tile size", style="CardLabel.TLabel").grid(row=3, column=0, sticky="w")
            ttk.Spinbox(
                realesrgan_frame,
                from_=0,
                to=1024,
                increment=32,
                textvariable=self._realesrgan_tile_var,
                width=8,
            ).grid(row=3, column=1, sticky="w", pady=(6, 6))
            ttk.Label(realesrgan_frame, text="Tile padding", style="CardLabel.TLabel").grid(row=3, column=2, sticky="w")
            ttk.Spinbox(
                realesrgan_frame,
                from_=0,
                to=256,
                increment=4,
                textvariable=self._realesrgan_tile_pad_var,
                width=8,
            ).grid(row=3, column=3, sticky="w", padx=(8, 0), pady=(6, 6))

            ttk.Label(advanced_card, text="ESRGAN strength (0-1)", style="CardLabel.TLabel").grid(
                row=2, column=0, sticky="w"
            )
            ttk.Spinbox(
                advanced_card,
                from_=0.0,
                to=1.0,
                increment=0.05,
                format="%.2f",
                textvariable=self._esrgan_strength_var,
                width=9,
            ).grid(row=2, column=1, sticky="w", pady=(6, 6))

            ttk.Label(advanced_card, text="ESRGAN detail limit", style="CardLabel.TLabel").grid(
                row=3, column=0, sticky="w"
            )
            ttk.Spinbox(
                advanced_card,
                from_=0.0,
                to=1.0,
                increment=0.05,
                format="%.2f",
                textvariable=self._esrgan_detail_limit_var,
                width=9,
            ).grid(row=3, column=1, sticky="w", pady=(6, 6))

            ttk.Label(advanced_card, text="Max pixels (0 = unlimited)", style="CardLabel.TLabel").grid(
                row=4, column=0, sticky="w"
            )
            ttk.Spinbox(
                advanced_card,
                from_=0,
                to=5000000,
                increment=100000,
                textvariable=self._max_pixels_var,
                width=11,
            ).grid(row=4, column=1, sticky="w", pady=(6, 6))

            ttk.Label(advanced_card, text="Texture extensions", style="CardLabel.TLabel").grid(
                row=5, column=0, sticky="w"
            )
            ttk.Combobox(
                advanced_card,
                textvariable=self._texture_exts_var,
                width=32,
                values=[
                    DEFAULT_TEXTURE_EXTENSIONS,
                    ".png,.jpg,.jpeg",
                    ".png,.dds,.tga",
                ],
            ).grid(row=5, column=1, sticky="ew", pady=(6, 6))

            ttk.Label(advanced_card, text="Character keywords (comma-separated)", style="CardLabel.TLabel").grid(
                row=6, column=0, sticky="w"
            )
            ttk.Combobox(
                advanced_card,
                textvariable=self._character_keywords_var,
                width=32,
                values=[
                    "",
                    "npc,character,enemy",
                    "boss,enemy,monster",
                ],
            ).grid(row=6, column=1, sticky="ew", pady=(6, 6))

            ttk.Label(advanced_card, text="Skip categories", style="CardLabel.TLabel").grid(
                row=7, column=0, sticky="w"
            )
            skip_button = ttk.Menubutton(
                advanced_card,
                textvariable=self._skip_categories_display,
                width=24,
                direction="below",
                style="Card.TMenubutton",
            )
            skip_menu = tk.Menu(skip_button, tearoff=False)
            for category, var in self._skip_category_vars.items():
                label = category.upper() if len(category) <= 2 else category.replace("_", " ").title()
                skip_menu.add_checkbutton(label=label, variable=var, command=self._update_skip_categories_selection)
            skip_button["menu"] = skip_menu
            skip_button.grid(row=7, column=1, sticky="w", pady=(6, 6))
            self._skip_categories_menu = skip_menu
            self._update_skip_categories_selection()

            self._build_about_tab(about_tab)

            log_frame = ttk.Frame(content, padding=(0, 12, 0, 12), style="TFrame")
            log_frame.grid(row=0, column=1, sticky="nsew", padx=0)
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(1, weight=1)
            self._log_frame = log_frame

            toolbar = ttk.Frame(log_frame, padding=(0, 0, 0, 8))
            toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
            toolbar.columnconfigure(0, weight=1)
            arcade_button = ttk.Button(
                toolbar,
                text="\u2728 Play Arcade",
                style="ArcadeLaunch.TButton",
                command=self._toggle_arcade_mode,
            )
            arcade_button.pack(side="right", padx=(12, 0))
            arcade_button._allow_during_run = True  # type: ignore[attr-defined]
            self._arcade_button = arcade_button

            log_box = tk.Text(
                log_frame,
                width=60,
                height=11,
                wrap="word",
                background="#11151d",
                foreground=TEXT_PRIMARY,
                insertbackground=ACCENT,
                relief="flat",
                borderwidth=0,
            )
            log_box.grid(row=1, column=0, sticky="nsew")
            log_box.configure(state="disabled")
            self._log_box = log_box

            scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_box.yview)
            scrollbar.grid(row=1, column=1, sticky="ns")
            log_box.configure(yscrollcommand=scrollbar.set)
            self._log_scrollbar = scrollbar

            arcade_container = ttk.Frame(log_frame, padding=0, style="TFrame")
            arcade_container.grid(row=1, column=0, columnspan=2, sticky="nsew")
            arcade_container.grid_remove()
            self._arcade_container = arcade_container

            toggle_container = ttk.Frame(content, padding=(0, 12, 12, 12), style="TFrame", width=32)
            toggle_container.grid(row=0, column=2, sticky="ns")
            toggle_container.rowconfigure(0, weight=1)
            toggle_container.grid_propagate(False)
            self._log_toggle_container = toggle_container

            toggle_button = ttk.Button(
                toggle_container,
                text="\u25B6",
                style="TerminalToggle.TButton",
                command=self._toggle_log_visibility,
                width=2,
            )
            toggle_button._allow_during_run = True  # type: ignore[attr-defined]
            toggle_button.pack(fill="both", expand=True)
            self._log_toggle_button = toggle_button

            self._set_log_visibility(self._log_visible.get())
            self._capture_config_controls()
            self.after(0, self._capture_initial_layout_widths)

        def _set_window_icon(self) -> None:
            if self._icon_photo is None:
                for root in _expand_asset_roots(self._resource_dir, self._exec_dir, Path.cwd()):
                    candidate = root / "upscaler_icon.png"
                    if candidate.is_file():
                        try:
                            self._icon_photo = tk.PhotoImage(file=str(candidate))
                            break
                        except Exception:
                            try:
                                with Image.open(candidate) as icon_image:
                                    self._icon_photo = ImageTk.PhotoImage(icon_image)
                                    break
                            except Exception as exc:
                                logging.debug("Unable to load window icon %s: %s", candidate, exc)
                if self._icon_photo is None:
                    logging.debug("Window icon asset not found; using default Tk icon.")
            if self._icon_photo is not None:
                try:
                    self.iconphoto(True, self._icon_photo)
                except Exception as exc:
                    logging.debug("Unable to set window icon: %s", exc)

        def _build_about_tab(self, container: ttk.Frame) -> None:
            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=1)

            wrapper = ttk.Frame(container, padding=20, style="Card.TFrame")
            wrapper.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
            wrapper.columnconfigure(0, weight=1)
            wrapper.rowconfigure(7, weight=1)

            ttk.Label(
                wrapper,
                text="DOOM Upscaler",
                style="CardLabel.TLabel",
                font=("Segoe UI", 14, "bold"),
            ).grid(row=0, column=0, sticky="n", pady=(0, 6))

            ttk.Label(
                wrapper,
                text="I am an indie developer crafting tools, mods, and retro-flavored games after hours.",
                style="CardLabel.TLabel",
                wraplength=360,
                justify="center",
            ).grid(row=1, column=0, sticky="n")

            ttk.Label(
                wrapper,
                text="If this upscaler saved you time or inspired a new project, consider tossing a little ETH my way so I can keep shipping weird ideas and soundtrack-fueled updates.",
                style="CardLabel.TLabel",
                wraplength=360,
                justify="center",
            ).grid(row=2, column=0, sticky="n", pady=(6, 0))

            ttk.Label(
                wrapper,
                text="Scan to tip the developer in ETH:",
                style="CardLabel.TLabel",
            ).grid(row=3, column=0, sticky="n", pady=(10, 0))

            qr_photo = self._get_eth_qr_photo()
            if qr_photo is not None:
                qr_label = tk.Label(wrapper, image=qr_photo, bg=CARD_BG)
                qr_label.image = qr_photo
                qr_label.grid(row=4, column=0, pady=(14, 10))
            else:
                ttk.Label(
                    wrapper,
                    text="(QR code unavailable)",
                    style="CardLabel.TLabel",
                    foreground=TEXT_SECONDARY,
                ).grid(row=4, column=0, pady=(14, 10))

            address_box = ttk.Frame(wrapper, padding=10, style="Card.TFrame")
            address_box.grid(row=5, column=0, sticky="ew")
            address_box.columnconfigure(0, weight=1)
            ttk.Label(
                address_box,
                text=ETH_DONATION_ADDRESS,
                style="CardLabel.TLabel",
                font=("Consolas", 11),
                anchor="center",
                justify="center",
            ).grid(row=0, column=0, sticky="ew")
            ttk.Button(
                address_box,
                text="Copy address",
                style="Card.TButton",
                command=self._copy_eth_address,
            ).grid(row=1, column=0, pady=(6, 0))

            ttk.Label(
                wrapper,
                text="Thank you for helping me keep the lights on and the experiments flowing!",
                style="Secondary.TLabel",
            ).grid(row=6, column=0, pady=(14, 0))

        def _copy_eth_address(self) -> None:
            try:
                self.clipboard_clear()
                self.clipboard_append(ETH_DONATION_ADDRESS)
                self._status_var.set("ETH address copied to clipboard.")
            except Exception as exc:  # noqa: BLE001 - clipboard failures aren't critical
                logging.debug("Unable to copy ETH address to clipboard: %s", exc)
                self._status_var.set("Unable to copy ETH address to clipboard.")

        def _select_file(self) -> None:
            filetypes = [
                ("WAD files", "*.wad"),
                ("PK3 files", "*.pk3"),
                ("All files", "*.*"),
            ]
            selected = filedialog.askopenfilename(title="Select texture archive", filetypes=filetypes)
            if selected:
                self._file_var.set(selected)

        def _launch_upscaler(self) -> None:
            if self._worker is not None:
                return
            input_path = self._file_var.get()
            if not input_path:
                messagebox.showwarning("DOOM Upscaler", "Pick a WAD or PK3 before running the upscaler.")
                return

            scale_value = max(1, int(round(float(self._scale_var.get()))))
            self._scale_var.set(float(scale_value))
            self._scale_display_var.set(f"{scale_value}x")

            command = self._select_upscaler_command(Path(input_path).resolve(), scale_value)
            if command is None:
                return

            self._abort_requested = False
            self._progress_processed = 0
            self._progress_total = 0
            self._append_log(f"[launcher] Running: {' '.join(shlex.quote(part) for part in command)}")
            self._status_var.set("Upscaling in progress...")
            self._set_running(True)

            self._worker = UpscaleWorker(command, self._log_queue)
            self._worker.start()

        def _abort_upscaler(self) -> None:
            if self._worker is None:
                return
            if self._abort_requested:
                return
            self._abort_requested = True
            self._status_var.set("Stopping current upscale...")
            if self._run_button:
                self._run_button.configure(text="Aborting...")
                self._run_button.state(["disabled"])
            try:
                self._worker.abort()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"[launcher] Failed to abort gracefully: {exc}")

        def _select_upscaler_command(self, input_path: Path, scale_value: int) -> Optional[list[str]]:
            extra_args = self._build_cli_args(input_path, scale_value)
            if extra_args is None:
                return None

            exe_candidate = self._exec_dir / "upscaler.exe"
            if exe_candidate.is_file():
                return [str(exe_candidate)] + extra_args

            script_candidates = [
                self._exec_dir / "upscaler.py",
                self._resource_dir / "upscaler.py",
            ]
            for script in script_candidates:
                if script.is_file():
                    return [sys.executable, str(script)] + extra_args

            messagebox.showerror(
                "DOOM Upscaler",
                (
                    "Could not locate upscaler executable or script. "
                    "Ensure the upscaler resides alongside this launcher."
                ),
            )
            return None

        def _set_running(self, running: bool) -> None:
            self._set_config_state(not running)
            if not self._run_button:
                return
            if running:
                self._run_button.configure(text="Abort", command=self._abort_upscaler)
                self._run_button.state(["!disabled"])
            else:
                self._run_button.configure(text="Enhance Textures", command=self._launch_upscaler)
                self._run_button.state(["!disabled"])

        def _toggle_log_visibility(self) -> None:
            self._set_log_visibility(not self._log_visible.get())

        def _set_log_visibility(self, visible: bool) -> None:
            if self._log_frame is None or self._log_toggle_button is None:
                return

            self._log_visible.set(visible)
            if visible:
                self._log_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16))
                self._log_frame.rowconfigure(1, weight=1)
                self._apply_log_or_arcade_view()
            else:
                self._pause_arcade()
                if self._log_box is not None:
                    self._log_box.grid_remove()
                if self._log_scrollbar is not None:
                    self._log_scrollbar.grid_remove()
                if self._arcade_container is not None:
                    self._arcade_container.grid_remove()
                self._log_frame.grid_forget()

            if self._content_frame is not None:
                self._content_frame.columnconfigure(1, weight=2 if visible else 0)

            self._log_toggle_button.configure(text="\u25C0" if visible else "\u25B6")
            self._adjust_window_width(visible)

        def _toggle_arcade_mode(self) -> None:
            self._set_arcade_mode(not self._arcade_mode.get())

        def _set_arcade_mode(self, enabled: bool) -> None:
            if self._arcade_mode.get() == enabled:
                if self._log_visible.get():
                    self._apply_log_or_arcade_view()
                return
            self._arcade_mode.set(enabled)
            if self._arcade_button is not None:
                self._arcade_button.configure(
                    text="\u2190 Show Terminal" if enabled else "\u2728 Play Arcade"
                )
            if not self._log_visible.get():
                self._pause_arcade()
                return
            self._apply_log_or_arcade_view()

        def _apply_log_or_arcade_view(self) -> None:
            if not self._log_visible.get():
                return
            arcade_active = self._arcade_mode.get()
            if arcade_active:
                if self._log_box is not None:
                    self._log_box.grid_remove()
                if self._log_scrollbar is not None:
                    self._log_scrollbar.grid_remove()
                if self._arcade_container is not None:
                    self._arcade_container.grid(row=1, column=0, columnspan=2, sticky="nsew")
                    if self._arcade_panel is None:
                        self._arcade_panel = self._ArcadePanel(
                            self._arcade_container,
                            exit_callback=self._on_arcade_exit,
                        )
                    self._arcade_panel.show()
            else:
                if self._arcade_container is not None:
                    self._arcade_container.grid_remove()
                self._pause_arcade()
                if self._log_box is not None:
                    self._log_box.grid(row=1, column=0, sticky="nsew")
                if self._log_scrollbar is not None:
                    self._log_scrollbar.grid(row=1, column=1, sticky="ns")

        def _pause_arcade(self) -> None:
            if self._arcade_panel is not None:
                self._arcade_panel.pause()

        def _on_arcade_exit(self) -> None:
            if self._arcade_mode.get():
                self._set_arcade_mode(False)

        class _ArcadePanel:
            @dataclass
            class _Asteroid:
                x: float
                y: float
                speed: float
                radius: float
                gold: bool
                points: List[Tuple[float, float]]
                counted: bool = False

            def __init__(
                self,
                container: tk.Widget,
                exit_callback: Optional[Callable[[], None]] = None,
            ) -> None:
                self.container = container
                self.canvas = tk.Canvas(
                    container,
                    width=ARCADE_WIDTH,
                    height=ARCADE_HEIGHT,
                    bg="#050b18",
                    highlightthickness=0,
                )
                self.canvas.pack(fill="both", expand=True)
                self.canvas.focus_set()
                self.canvas.bind("<Button-1>", self._on_canvas_click)
                self.canvas.bind("<KeyPress>", self._on_key_press)
                self.canvas.bind("<KeyRelease>", self._on_key_release)
                self.canvas.bind("<Destroy>", lambda _e: self.pause())

                self._after_id: Optional[str] = None
                self._running = False
                self._keys: Set[str] = set()
                self._scoreboard_active = False
                self._stars = self._init_stars()
                self._credits_entries = self._build_credits_entries()
                self._scale = 1.0
                self._offset_x = 0.0
                self._offset_y = 0.0
                self._font_cache: Dict[Tuple[str, int, str], Tuple[Any, ...]] = {}
                self._paused = False
                self._exit_callback = exit_callback
                self._menu_selection = 0
                self._menu_button_regions: List[Tuple[int, Tuple[float, float, float, float]]] = []
                self._menu_particles: List[Dict[str, float]] = []
                self._menu_pulse = 0.0
                self._menu_options: List[Tuple[str, str]] = [
                    ("Play Game", "play"),
                    ("Score Board", "score"),
                    ("Quit", "quit"),
                ]

                self._ship_x = ARCADE_SHIP_LEFT_BOUND + 20
                self._ship_y = ARCADE_HEIGHT / 2
                self._ship_radius = ARCADE_SHIP_HEIGHT // 3
                self._ship_half_width = 32
                self._ship_half_height = 20

                self._reset_game_state()
                self.show()

            def show(self) -> None:
                if self._running:
                    return
                self._running = True
                self._last_tick = time.perf_counter()
                self._schedule_tick()

            def pause(self) -> None:
                if self._after_id is not None:
                    self.canvas.after_cancel(self._after_id)
                    self._after_id = None
                self._running = False

            def _update_projection(self) -> None:
                width = max(self.canvas.winfo_width(), 1)
                height = max(self.canvas.winfo_height(), 1)
                if width <= 2 and height <= 2:
                    width = max(self.canvas.winfo_reqwidth(), ARCADE_WIDTH)
                    height = max(self.canvas.winfo_reqheight(), ARCADE_HEIGHT)
                prev_scale = self._scale
                new_scale = min(width / ARCADE_WIDTH, height / ARCADE_HEIGHT)
                if not math.isfinite(new_scale) or new_scale <= 0:
                    new_scale = prev_scale if prev_scale > 0 else 1.0
                if not math.isclose(new_scale, prev_scale, rel_tol=1e-4):
                    self._font_cache.clear()
                self._scale = new_scale
                self._offset_x = (width - ARCADE_WIDTH * self._scale) / 2
                self._offset_y = (height - ARCADE_HEIGHT * self._scale) / 2

            def _project(self, x: float, y: float) -> Tuple[float, float]:
                return x * self._scale + self._offset_x, y * self._scale + self._offset_y

            def _unproject(self, sx: float, sy: float) -> Tuple[float, float]:
                if self._scale <= 0:
                    return sx, sy
                return (sx - self._offset_x) / self._scale, (sy - self._offset_y) / self._scale

            def _rect_coords(self, left: float, top: float, right: float, bottom: float) -> Tuple[float, float, float, float]:
                x0, y0 = self._project(left, top)
                x1, y1 = self._project(right, bottom)
                return x0, y0, x1, y1

            def _scale_value(self, value: float) -> float:
                return value * self._scale

            def _line_width(self, width: float) -> float:
                return max(1.0, width * self._scale)

            def _poly_coords(self, points: Iterable[Tuple[float, float]]) -> List[float]:
                coords: List[float] = []
                for px, py in points:
                    sx, sy = self._project(px, py)
                    coords.extend((sx, sy))
                return coords

            def _font(self, size: int, style: Optional[str] = None, *, family: str = FONT_NAME) -> tuple[Any, ...]:
                normalized = (style or "").strip().lower()
                if normalized == "normal":
                    normalized = ""
                scaled_size = max(6, int(round(size * self._scale)))
                key = (family, scaled_size, normalized)
                font = self._font_cache.get(key)
                if font is None:
                    if normalized:
                        font = (family, scaled_size, normalized)
                    else:
                        font = (family, scaled_size)
                    self._font_cache[key] = font
                return font

            def _reset_game_state(self) -> None:
                self._asteroids: List[UpscalerApp._ArcadePanel._Asteroid] = []
                self._score = 0
                self._lives = INITIAL_LIVES
                self._elapsed_ms = 0.0
                self._spawn_ms = 0.0
                self._landing_elapsed = 0.0
                self._message_timer = 0.0
                self._credits_offset = ARCADE_HEIGHT + 120
                self._credits_looped = False
                self._planet_center = (ARCADE_WIDTH - 140, ARCADE_HEIGHT / 2)
                self._planet_radius = 140
                self._continents = self._generate_continents()
                self._last_tick = time.perf_counter()
                self._paused = False
                self._scoreboard_active = False
                self._keys.clear()
                self._menu_selection = 0
                self._menu_button_regions = []
                self._menu_particles = self._create_menu_particles()
                self._menu_pulse = 0.0
                self._state = "menu"

            def _start_gameplay(self) -> None:
                self._asteroids = []
                self._score = 0
                self._lives = INITIAL_LIVES
                self._elapsed_ms = 0.0
                self._spawn_ms = 0.0
                self._landing_elapsed = 0.0
                self._message_timer = 0.0
                self._credits_offset = ARCADE_HEIGHT + 120
                self._credits_looped = False
                self._planet_center = (ARCADE_WIDTH - 140, ARCADE_HEIGHT / 2)
                self._planet_radius = 140
                self._continents = self._generate_continents()
                self._keys.clear()
                self._state = "playing"

            def _generate_continents(self) -> List[List[Tuple[float, float]]]:
                continents: List[List[Tuple[float, float]]] = []
                count = random.randint(4, 6)
                max_extent = 0.95
                angular_jitter = math.pi / (count * 1.8)
                for index in range(count):
                    major = random.uniform(0.18, 0.32)
                    minor = major * random.uniform(0.45, 0.8)
                    base_angle = 2 * math.pi * index / count
                    angle = base_angle + random.uniform(-angular_jitter, angular_jitter)
                    usable_span = max_extent - max(major, minor)
                    span = max(0.05, usable_span)
                    distance = 0.05 + math.pow(random.random(), 0.4) * (span - 0.05)
                    center_x = math.cos(angle) * distance
                    center_y = math.sin(angle) * distance * random.uniform(0.65, 1.0)
                    center_x = max(-max_extent + major, min(max_extent - major, center_x))
                    center_y = max(-max_extent + minor, min(max_extent - minor, center_y))
                    rotation = random.uniform(-math.pi / 2, math.pi / 2)
                    point_count = random.randint(16, 24)
                    jagged = random.uniform(0.05, 0.15)
                    points: List[Tuple[float, float]] = []
                    cos_rot = math.cos(rotation)
                    sin_rot = math.sin(rotation)
                    for i in range(point_count):
                        theta = 2 * math.pi * i / point_count
                        noise = 1.0 + random.uniform(-jagged, jagged)
                        px = math.cos(theta) * major * noise
                        py = math.sin(theta) * minor * noise
                        rx = px * cos_rot - py * sin_rot
                        ry = px * sin_rot + py * cos_rot
                        points.append((center_x + rx, center_y + ry))
                    continents.append(points)
                return continents
    
            def _create_menu_particles(self) -> List[Dict[str, float]]:
                particles: List[Dict[str, float]] = []
                for _ in range(28):
                    particles.append(
                        {
                            "x": random.uniform(0, ARCADE_WIDTH),
                            "y": random.uniform(0, ARCADE_HEIGHT),
                            "speed": random.uniform(18.0, 42.0),
                            "size": random.uniform(10.0, 26.0),
                            "phase": random.uniform(0.0, 2 * math.pi),
                        }
                    )
                return particles

            def _update_menu_animation(self, dt: float) -> None:
                if not self._menu_particles:
                    self._menu_particles = self._create_menu_particles()
                seconds = dt / 1000.0
                for particle in self._menu_particles:
                    particle["x"] -= particle["speed"] * seconds
                    particle["phase"] += seconds * 2.0
                    if particle["x"] < -40:
                        particle["x"] = ARCADE_WIDTH + random.uniform(0, 60)
                        particle["y"] = random.uniform(0, ARCADE_HEIGHT)
                        particle["speed"] = random.uniform(18.0, 42.0)
                        particle["size"] = random.uniform(10.0, 26.0)
                        particle["phase"] = random.uniform(0.0, 2 * math.pi)
                self._menu_pulse = (self._menu_pulse + seconds * 3.0) % (2 * math.pi)

            def _schedule_tick(self) -> None:
                if not self._running:
                    return
                self._tick()
                self._after_id = self.canvas.after(int(1000 / ARCADE_FPS), self._schedule_tick)
    
            def _tick(self) -> None:
                now = time.perf_counter()
                dt = min((now - getattr(self, "_last_tick", now)) * 1000.0, 100.0)
                self._last_tick = now
                if self._scoreboard_active:
                    self._draw()
                    return
                if self._paused:
                    self._draw()
                    return
                if self._state == "menu":
                    self._update_menu_animation(dt)
                elif self._state == "playing":
                    self._update_game(dt)
                elif self._state == "landing":
                    self._update_landing(dt)
                elif self._state == "message":
                    self._update_success_message(dt)
                elif self._state == "credits":
                    self._update_credits(dt)
                self._update_stars(dt)
                self._draw()
    
            def _update_game(self, dt: float) -> None:
                self._elapsed_ms += dt
                intensity = min(1.0, self._elapsed_ms / ARCADE_SURVIVAL_MS)
                interval = lerp(ARCADE_SPAWN_INTERVAL_START_MS, ARCADE_SPAWN_INTERVAL_END_MS, intensity)
                self._spawn_ms += dt
                while self._spawn_ms >= interval:
                    self._spawn_ms -= interval
                    self._asteroids.append(self._create_asteroid(intensity))
                self._update_ship(dt)
                self._update_asteroids(dt)
                self._check_collisions()
                if self._elapsed_ms >= ARCADE_SURVIVAL_MS:
                    record_arcade_score(self._score)
                    self._state = "landing"
                    self._landing_elapsed = 0.0
                    self._landing_travel = 4500.0
                    self._landing_settle = 2500.0
                    self._landing_start_ship = (self._ship_x, self._ship_y)
    
            def _update_ship(self, dt: float) -> None:
                mult = dt * ARCADE_FPS / 1000.0
                if "up" in self._keys:
                    self._ship_y -= ARCADE_SHIP_SPEED * mult
                if "down" in self._keys:
                    self._ship_y += ARCADE_SHIP_SPEED * mult
                if "left" in self._keys:
                    self._ship_x -= ARCADE_FORWARD_SPEED * mult
                if "right" in self._keys:
                    self._ship_x += ARCADE_FORWARD_SPEED * mult
    
                self._ship_y = max(self._ship_radius + 4, min(ARCADE_HEIGHT - self._ship_radius - 4, self._ship_y))
                min_x = ARCADE_SHIP_LEFT_BOUND
                max_x = ARCADE_WIDTH * ARCADE_SHIP_RIGHT_RATIO
                self._ship_x = max(min_x, min(max_x, self._ship_x))
    
            def _update_asteroids(self, dt: float) -> None:
                mult = dt * ARCADE_FPS / 1000.0
                for asteroid in list(self._asteroids):
                    asteroid.x -= asteroid.speed * mult
                    if not asteroid.counted and asteroid.x + asteroid.radius < self._ship_x - self._ship_half_width:
                        asteroid.counted = True
                        self._score += ARCADE_POINTS_GOLD if asteroid.gold else ARCADE_POINTS_NORMAL
                    if asteroid.x + asteroid.radius < -10:
                        self._asteroids.remove(asteroid)
    
            def _check_collisions(self) -> None:
                for asteroid in list(self._asteroids):
                    dx = asteroid.x - self._ship_x
                    dy = asteroid.y - self._ship_y
                    limit = asteroid.radius + self._ship_radius
                    if dx * dx + dy * dy <= limit * limit:
                        self._asteroids.remove(asteroid)
                        if asteroid.gold:
                            self._score += ARCADE_POINTS_GOLD
                        else:
                            self._score = max(0, self._score - ARCADE_HIT_PENALTY)
                            self._lives -= 1
                            if self._lives <= 0:
                                self._state = "gameover"
                                record_arcade_score(self._score)
                        break
    
            def _create_asteroid(self, intensity: float) -> "_Asteroid":
                size_min = ASTEROID_MIN_SIZE
                size_max = ASTEROID_MAX_SIZE + int(ASTEROID_SIZE_BONUS * intensity)
                size = random.randint(size_min, size_max)
                speed_min = ASTEROID_MIN_SPEED + ASTEROID_SPEED_BONUS * intensity * 0.4
                speed_max = ASTEROID_MAX_SPEED + ASTEROID_SPEED_BONUS * intensity
                speed = random.uniform(speed_min, speed_max)
                gold = random.random() < min(0.6, GOLD_ASTEROID_CHANCE + GOLD_BONUS_EXTRA * intensity)
                y = random.randint(size, ARCADE_HEIGHT - size)
                radius = size / 2
                points: List[Tuple[float, float]] = []
                spikes = random.randint(5, 9)
                for i in range(spikes):
                    offset = random.uniform(0.65, 1.0)
                    theta = (2 * math.pi / spikes) * i
                    px = math.cos(theta) * radius * offset
                    py = math.sin(theta) * radius * offset
                    points.append((px, py))
                return self._Asteroid(x=ARCADE_WIDTH + radius + 20, y=y, speed=speed, radius=radius, gold=gold, points=points)
    
            def _update_landing(self, dt: float) -> None:
                self._landing_elapsed += dt
                travel = self._landing_travel
                settle = self._landing_settle
                if self._landing_elapsed <= travel:
                    t = self._landing_elapsed / travel
                    self._planet_center = (lerp(ARCADE_WIDTH + 150, ARCADE_WIDTH - 140, t), ARCADE_HEIGHT / 2)
                    self._planet_radius = lerp(60, 140, t)
                    self._ship_x = lerp(self._landing_start_ship[0], self._planet_center[0] - self._planet_radius - 30, t)
                    self._ship_y = ARCADE_HEIGHT * 0.4 + math.sin(t * math.pi) * 30
                elif self._landing_elapsed <= travel + settle:
                    inner = self._landing_elapsed - travel
                    t = inner / settle
                    self._planet_center = (
                        lerp(ARCADE_WIDTH - 140, ARCADE_WIDTH / 2 + 120, t),
                        lerp(ARCADE_HEIGHT / 2, ARCADE_HEIGHT - 180, t),
                    )
                    self._planet_radius = lerp(140, 220, t)
                    self._ship_x = lerp(self._planet_center[0] - self._planet_radius - 30, self._planet_center[0] - 40, t)
                    self._ship_y = lerp(ARCADE_HEIGHT * 0.4, ARCADE_HEIGHT - self._planet_radius - 10, t)
                else:
                    self._state = "message"
                    self._message_timer = 2000.0
    
            def _update_success_message(self, dt: float) -> None:
                self._message_timer -= dt
                if self._message_timer <= 0:
                    self._state = "credits"
                    self._credits_offset = ARCADE_HEIGHT + 120
                    self._credits_looped = False
    
            def _update_credits(self, dt: float) -> None:
                scroll_speed = 140.0
                self._credits_offset -= scroll_speed * (dt / 1000.0)
                total_height = len(self._credits_entries) * 42
                if self._credits_offset + total_height < -120:
                    self._credits_offset = ARCADE_HEIGHT + 120
                    self._credits_looped = True
    
            def _update_stars(self, dt: float) -> None:
                speed = 0.12 * dt
                for star in self._stars:
                    star[0] -= star[3] * speed
                    if star[0] < -5:
                        star[0] = ARCADE_WIDTH + random.randint(0, 40)
                        star[1] = random.randint(0, ARCADE_HEIGHT)
                        star[3] = random.uniform(1.0, 3.0)
    
            def _draw(self) -> None:
                self._update_projection()
                c = self.canvas
                c.delete("all")
                self._draw_background()
                if self._state == "menu":
                    self._draw_menu()
                else:
                    if self._state in {"landing", "message", "credits"}:
                        self._draw_planet()
                    self._draw_ship()
                    if self._state == "playing":
                        self._draw_asteroids()
                    elif self._state == "gameover":
                        self._draw_asteroids()
                        self._draw_game_over_overlay()
                    elif self._state == "message":
                        self._draw_success_message()
                    elif self._state == "credits":
                        self._draw_credits()
                    if self._state not in {"menu", "credits"}:
                        self._draw_hud()
                if self._scoreboard_active:
                    self._draw_scoreboard_overlay()
                elif self._paused:
                    self._draw_pause_overlay()
    
            def _draw_background(self) -> None:
                for x, y, size, _speed in self._stars:
                    color = "#9fb0ff" if size < 1.5 else "#d8e3ff"
                    sx, sy = self._project(x, y)
                    radius = self._scale_value(size)
                    self.canvas.create_oval(sx, sy, sx + radius, sy + radius, fill=color, outline="")
    
            def _draw_menu(self) -> None:
                self._menu_button_regions = []
                overlay = self._rect_coords(0, 0, ARCADE_WIDTH, ARCADE_HEIGHT)
                self.canvas.create_rectangle(*overlay, fill="#050b18", stipple="gray25", outline="")
                for particle in self._menu_particles:
                    px, py = self._project(particle["x"], particle["y"])
                    size = self._scale_value(particle["size"])
                    hue = 0.45 + 0.3 * math.sin(particle["phase"])
                    color = "#3b4f78" if hue < 0.5 else "#4cb0ff"
                    self.canvas.create_oval(
                        px - size / 2,
                        py - size / 2,
                        px + size / 2,
                        py + size / 2,
                        fill=color,
                        outline="",
                    )
                title_x, title_y = self._project(ARCADE_WIDTH / 2, 140)
                pulse = 0.2 * math.sin(self._menu_pulse) + 0.8
                self.canvas.create_text(
                    title_x,
                    title_y,
                    text="Gold Star",
                    fill="#ffe07a",
                    font=self._font(int(42 * pulse), "bold"),
                    anchor="center",
                )
                subtitle_x, subtitle_y = self._project(ARCADE_WIDTH / 2, 190)
                self.canvas.create_text(
                    subtitle_x,
                    subtitle_y,
                    text="Asteroid Mining Rush",
                    fill="#b8c7ff",
                    font=self._font(20),
                    anchor="center",
                )
                button_width = 280
                button_height = 50
                spacing = 18
                start_y = ARCADE_HEIGHT / 2 - 20
                for idx, (label, _action) in enumerate(self._menu_options):
                    top = start_y + idx * (button_height + spacing)
                    left = ARCADE_WIDTH / 2 - button_width / 2
                    right = left + button_width
                    bottom = top + button_height
                    selected = idx == self._menu_selection
                    fill = "#14203a" if not selected else "#1f2f52"
                    outline = "#22324f" if not selected else "#46f0ff"
                    rect = self._rect_coords(left, top, right, bottom)
                    self.canvas.create_rectangle(
                        *rect,
                        fill=fill,
                        outline=outline,
                        width=self._line_width(2 if selected else 1),
                    )
                    text_color = "#f2f6ff" if selected else "#c6d7ff"
                    tx, ty = self._project(ARCADE_WIDTH / 2, top + button_height / 2)
                    self.canvas.create_text(
                        tx,
                        ty,
                        text=label,
                        fill=text_color,
                        font=self._font(18, "bold" if selected else None),
                        anchor="center",
                    )
                    self._menu_button_regions.append((idx, (left, top, right, bottom)))
                hint_x, hint_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT - 60)
                self.canvas.create_text(
                    hint_x,
                    hint_y,
                    text="Use Arrow Keys or Click • Enter to confirm",
                    fill="#7fa0ff",
                    font=self._font(12),
                    anchor="center",
                )
    
            def _draw_ship(self) -> None:
                ship_width = 100.0
                ship_height = 60.0
                origin_x = self._ship_x - ship_width / 2
                origin_y = self._ship_y - ship_height / 2
    
                def rect_local(left: float, top: float, width: float, height: float) -> Tuple[float, float, float, float]:
                    return self._rect_coords(
                        origin_x + left,
                        origin_y + top,
                        origin_x + left + width,
                        origin_y + top + height,
                    )
    
                hull_dark = "#284678"
                hull_main = "#4896eb"
                tail_left = 8.0
                tail_top = 16.0
                tail_width = ship_width - 26.0
                tail_height = ship_height - 32.0
                tail_center_x = tail_left + tail_width / 2
                tail_center_y = tail_top + tail_height / 2
    
                self.canvas.create_oval(
                    *rect_local(tail_left, tail_top, tail_width, tail_height),
                    fill=hull_dark,
                    outline="#142943",
                    width=self._line_width(2),
                )
                self.canvas.create_oval(
                    *rect_local(tail_left + 6, tail_top + 6, tail_width - 12, tail_height - 12),
                    fill=hull_main,
                    outline="",
                )
    
                stripe_left = tail_left + 10
                stripe_width = tail_width - 30
                stripe_height = 10
                stripe_top = tail_center_y - stripe_height / 2
                self.canvas.create_rectangle(
                    *rect_local(stripe_left, stripe_top, stripe_width, stripe_height),
                    fill="#f7fbff",
                    outline="",
                )
    
                canopy_width = 30
                canopy_height = 20
                canopy_left = tail_center_x - canopy_width / 2 - 2
                canopy_top = tail_center_y - canopy_height / 2
                self.canvas.create_oval(
                    *rect_local(canopy_left, canopy_top, canopy_width, canopy_height),
                    fill="#c2e7ff",
                    outline="#ffffff",
                    width=self._line_width(1),
                )
                highlight_inset_x = 10
                highlight_inset_y = 12
                self.canvas.create_oval(
                    *rect_local(
                        canopy_left + highlight_inset_x / 2,
                        canopy_top + highlight_inset_y / 2,
                        canopy_width - highlight_inset_x,
                        canopy_height - highlight_inset_y,
                    ),
                    outline="#ffffff",
                    width=self._line_width(1),
                )
    
                def draw_thruster(y_offset: float) -> None:
                    pod_width = 48
                    pod_height = 12
                    pod_left = tail_center_x - pod_width / 2
                    pod_top = y_offset
                    self.canvas.create_oval(
                        *rect_local(pod_left, pod_top, pod_width, pod_height),
                        fill=hull_dark,
                        outline="#142943",
                        width=self._line_width(1),
                    )
                    self.canvas.create_oval(
                        *rect_local(pod_left + 6, pod_top + 2, pod_width - 12, pod_height - 4),
                        fill="#506aa8",
                        outline="",
                    )
                    glow_left = pod_left - 12
                    glow_width = 12
                    glow_height = pod_height - 4
                    glow_top = pod_top + 2
                    self.canvas.create_rectangle(
                        *rect_local(glow_left, glow_top, glow_width, glow_height),
                        fill="#ffd28c",
                        outline="",
                    )
                    plume_left = glow_left - 8
                    plume_width = 8
                    plume_height = glow_height - 2
                    plume_top = glow_top + 1
                    self.canvas.create_rectangle(
                        *rect_local(plume_left, plume_top, plume_width, plume_height),
                        fill="#fff3d5",
                        outline="",
                    )
    
                draw_thruster(tail_top - 10)
                draw_thruster(tail_top + tail_height - 2)
    
            def _draw_asteroids(self) -> None:
                for asteroid in self._asteroids:
                    coords = self._poly_coords(( (asteroid.x + px, asteroid.y + py) for px, py in asteroid.points ))
                    color = "#ffd86a" if asteroid.gold else "#8c8898"
                    self.canvas.create_polygon(
                        *coords,
                        fill=color,
                        outline="#1d1833",
                        width=self._line_width(2),
                    )
    
            def _draw_planet(self) -> None:
                cx, cy = self._planet_center
                r = self._planet_radius
                self.canvas.create_oval(
                    *self._rect_coords(cx - r, cy - r, cx + r, cy + r),
                    fill="#1d71f2",
                    outline="#0f3f91",
                    width=self._line_width(2),
                )
                continents = getattr(self, "_continents", None)
                if not continents:
                    continents = self._generate_continents()
                    self._continents = continents
                for polygon in continents:
                    coords = self._poly_coords(
                        ((cx + px * r, cy + py * r) for px, py in polygon)
                    )
                    self.canvas.create_polygon(
                        *coords,
                        fill="#2bc15e",
                        outline="#166733",
                        width=self._line_width(1),
                        smooth=True,
                        splinesteps=12,
                    )
    
            def _draw_hud(self) -> None:
                x, y = self._project(ARCADE_WIDTH - 20, 24)
                self.canvas.create_text(
                    x,
                    y,
                    text=f"Score: {self._score:,}",
                    fill="#f5f5f5",
                    anchor="ne",
                    font=self._font(14, "bold"),
                )
                hearts = "\u2665" * max(0, self._lives)
                hx, hy = self._project(20, 20)
                self.canvas.create_text(
                    hx,
                    hy,
                    text=f"Lives: {hearts}",
                    fill="#ffaaaa",
                    anchor="nw",
                    font=self._font(14, "bold"),
                )
                progress = int((self._elapsed_ms / ARCADE_SURVIVAL_MS) * 100)
                px, py = self._project(ARCADE_WIDTH / 2, 18)
                self.canvas.create_text(
                    px,
                    py,
                    text=f"Progress: {progress:02d}%",
                    fill="#b2c7ff",
                    anchor="n",
                    font=self._font(13, "bold"),
                )
                total_seconds = int(self._elapsed_ms / 1000)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                tx, ty = self._project(ARCADE_WIDTH / 2, 38)
                self.canvas.create_text(
                    tx,
                    ty,
                    text=f"{minutes:02d}:{seconds:02d}",
                    fill="#a0a0a0",
                    anchor="n",
                    font=self._font(12),
                )
    
            def _draw_game_over_overlay(self) -> None:
                self.canvas.create_rectangle(
                    *self._rect_coords(40, ARCADE_HEIGHT / 2 - 80, ARCADE_WIDTH - 40, ARCADE_HEIGHT / 2 + 70),
                    fill="#050b18",
                    outline="#5c6dbb",
                    width=self._line_width(2),
                )
                title_x, title_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 - 30)
                self.canvas.create_text(
                    title_x,
                    title_y,
                    text="Ship Destroyed",
                    fill="#ffffff",
                    font=self._font(20, "bold"),
                )
                score_x, score_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 + 10)
                self.canvas.create_text(
                    score_x,
                    score_y,
                    text=f"Final Score: {self._score:,}",
                    fill="#d9ffdc",
                    font=self._font(16),
                )
                prompt_x, prompt_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 + 48)
                self.canvas.create_text(
                    prompt_x,
                    prompt_y,
                    text="Press Enter to play again | Space for score board",
                    fill="#cfd3ff",
                    font=self._font(12),
                )
    
            def _draw_success_message(self) -> None:
                self._draw_planet()
                self.canvas.create_rectangle(
                    *self._rect_coords(40, ARCADE_HEIGHT / 2 - 60, ARCADE_WIDTH - 40, ARCADE_HEIGHT / 2 + 40),
                    fill="#050b18",
                    outline="#5c6dbb",
                    width=self._line_width(2),
                )
                title_x, title_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 - 10)
                self.canvas.create_text(
                    title_x,
                    title_y,
                    text="You did it!",
                    fill="#ffffff",
                    font=self._font(20, "bold"),
                )
                score_x, score_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 + 20)
                self.canvas.create_text(
                    score_x,
                    score_y,
                    text=f"Final Score: {self._score:,}",
                    fill="#d9ffdc",
                    font=self._font(16),
                )
    
            def _draw_credits(self) -> None:
                title_x, title_y = self._project(ARCADE_WIDTH / 2, 60)
                self.canvas.create_text(
                    title_x,
                    title_y,
                    text="Mission Credits (They Keep Coming)",
                    fill="#ffffff",
                    font=self._font(18, "bold"),
                )
                score_x, score_y = self._project(ARCADE_WIDTH / 2, 90)
                self.canvas.create_text(
                    score_x,
                    score_y,
                    text=f"Final Score: {self._score:,}",
                    fill="#d7e0ff",
                    font=self._font(14),
                )
                spacing = 42
                start = int(max(0, (-self._credits_offset) // spacing) - 1)
                end = min(len(self._credits_entries), start + int(ARCADE_HEIGHT / spacing) + 6)
                for idx in range(start, end):
                    y = self._credits_offset + idx * spacing
                    if -60 <= y <= ARCADE_HEIGHT + 60:
                        row_x, row_y = self._project(ARCADE_WIDTH / 2, y)
                        self.canvas.create_text(
                            row_x,
                            row_y,
                            text=f"{self._credits_entries[idx]}: SteveDeFacto",
                            fill="#bcd4ff",
                            font=self._font(13),
                        )
                prompt = (
                    "Enter = Replay | Space = Score Board | Esc = Quit"
                    if self._credits_looped
                    else "Enter = Escape Credits | Space = Score Board"
                )
                prompt_x, prompt_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT - 40)
                self.canvas.create_text(
                    prompt_x,
                    prompt_y,
                    text=prompt,
                    fill="#ffffff",
                    font=self._font(12),
                )
    
            def _draw_scoreboard_overlay(self) -> None:
                self.canvas.create_rectangle(
                    *self._rect_coords(80, 60, ARCADE_WIDTH - 80, ARCADE_HEIGHT - 60),
                    fill="#000000",
                    stipple="gray50",
                    outline="#8ea8ff",
                    width=self._line_width(2),
                )
                title_x, title_y = self._project(ARCADE_WIDTH / 2, 90)
                self.canvas.create_text(
                    title_x,
                    title_y,
                    text="Score Board",
                    fill="#ffffff",
                    font=self._font(18, "bold"),
                )
                if ARCADE_HIGH_SCORES:
                    for idx, value in enumerate(ARCADE_HIGH_SCORES, start=1):
                        row_x, row_y = self._project(ARCADE_WIDTH / 2, 120 + idx * 28)
                        style = "bold" if idx == 1 else None
                        self.canvas.create_text(
                            row_x,
                            row_y,
                            text=f"{idx:02d}. {value:,}",
                            fill="#e0e6ff",
                            font=self._font(13, style),
                        )
                else:
                    empty_x, empty_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2)
                    self.canvas.create_text(
                        empty_x,
                        empty_y,
                        text="No recorded runs yet.",
                        fill="#d0d7ff",
                        font=self._font(13),
                    )
                prompt_x, prompt_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT - 80)
                self.canvas.create_text(
                    prompt_x,
                    prompt_y,
                    text="Press Enter, Space, or Esc to close",
                    fill="#ffffff",
                    font=self._font(12),
                )
    
            def _draw_pause_overlay(self) -> None:
                self.canvas.create_rectangle(
                    *self._rect_coords(60, 80, ARCADE_WIDTH - 60, ARCADE_HEIGHT - 80),
                    fill="#050b18",
                    stipple="gray50",
                    outline="#5c6dbb",
                    width=self._line_width(2),
                )
                title_x, title_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 - 20)
                self.canvas.create_text(
                    title_x,
                    title_y,
                    text="Paused",
                    fill="#ffffff",
                    font=self._font(20, "bold"),
                )
                prompt_x, prompt_y = self._project(ARCADE_WIDTH / 2, ARCADE_HEIGHT / 2 + 20)
                self.canvas.create_text(
                    prompt_x,
                    prompt_y,
                    text="Press Esc to resume | Space = Score Board",
                    fill="#d0d7ff",
                    font=self._font(12),
                )
    
            def _on_key_press(self, event: tk.Event) -> None:
                key = event.keysym.lower()
                if self._scoreboard_active:
                    if key in ("space", "return", "kp_enter", "escape"):
                        self._scoreboard_active = False
                    return
                if self._state == "menu":
                    if key in ("up", "w"):
                        self._change_menu_selection(-1)
                    elif key in ("down", "s"):
                        self._change_menu_selection(1)
                    elif key in ("return", "kp_enter", "space"):
                        self._activate_menu_option()
                    elif key in ("escape", "q"):
                        self._quit_to_terminal()
                    return
                if key == "escape":
                    self._paused = not self._paused
                    if self._paused:
                        self._keys.clear()
                    return
                if self._paused:
                    if key == "space":
                        self._scoreboard_active = True
                    return
                if self._state == "playing":
                    if key in ("up", "w"):
                        self._keys.add("up")
                    if key in ("down", "s"):
                        self._keys.add("down")
                    if key in ("left", "a"):
                        self._keys.add("left")
                    if key in ("right", "d"):
                        self._keys.add("right")
                    if key == "space":
                        self._scoreboard_active = True
                    return
                if key == "space":
                    self._scoreboard_active = True
                    return
                if key in ("return", "kp_enter"):
                    if self._state == "gameover":
                        self._reset_game_state()
                    elif self._state in {"landing", "message"}:
                        self._state = "credits"
                        self._credits_offset = ARCADE_HEIGHT + 120
                        self._credits_looped = False
                    elif self._state == "credits":
                        self._reset_game_state()
    
            def _on_key_release(self, event: tk.Event) -> None:
                key = event.keysym.lower()
                mapping = {"up": "up", "w": "up", "down": "down", "s": "down", "left": "left", "a": "left", "right": "right", "d": "right"}
                mapped = mapping.get(key)
                if mapped and mapped in self._keys:
                    self._keys.discard(mapped)
    
            def _on_canvas_click(self, event: tk.Event) -> None:
                self.canvas.focus_set()
                if self._state != "menu":
                    return
                if not self._menu_button_regions:
                    return
                x, y = self._unproject(event.x, event.y)
                for stored_index, (left, top, right, bottom) in self._menu_button_regions:
                    if left <= x <= right and top <= y <= bottom:
                        self._menu_selection = stored_index
                        self._activate_menu_option(stored_index)
                        return

            def _change_menu_selection(self, delta: int) -> None:
                if not self._menu_options:
                    self._menu_selection = 0
                    return
                count = len(self._menu_options)
                self._menu_selection = (self._menu_selection + delta) % count

            def _activate_menu_option(self, index: Optional[int] = None) -> None:
                if not self._menu_options:
                    return
                if index is None:
                    index = self._menu_selection
                index = max(0, min(len(self._menu_options) - 1, index))
                self._menu_selection = index
                _label, action = self._menu_options[index]
                if action == "play":
                    self._start_gameplay()
                elif action == "score":
                    self._scoreboard_active = True
                elif action == "quit":
                    self._quit_to_terminal()

            def _quit_to_terminal(self) -> None:
                if self._exit_callback is not None:
                    try:
                        self._exit_callback()
                    except Exception as exc:  # noqa: BLE001
                        logging.debug("Arcade exit callback failed: %s", exc)

            def _init_stars(self) -> List[List[float]]:
                stars: List[List[float]] = []
                for _ in range(90):
                    stars.append(
                        [
                            random.uniform(0, ARCADE_WIDTH),
                            random.uniform(0, ARCADE_HEIGHT),
                            random.uniform(1.0, 2.5),
                            random.uniform(1.0, 3.0),
                        ]
                    )
                return stars
    
            def _build_credits_entries(self) -> List[str]:
                adjective_pool = [
                    "Galactic",
                    "Hyperdrive",
                    "Quantum",
                    "Snack",
                    "Chaos",
                    "Orbit",
                    "Nebula",
                    "Temporal",
                    "Laser",
                    "Cosmic",
                    "Lunar",
                    "Solar",
                    "Plasma",
                    "Zero-G",
                    "Gravity",
                    "Void",
                    "Stellar",
                    "Meteor",
                    "Flux",
                    "Retro",
                ]
                middle_pool = [
                    "Asteroid",
                    "Mood",
                    "Logistics",
                    "Confetti",
                    "Etiquette",
                    "Vibes",
                    "Whisper",
                    "Protocol",
                    "Whale",
                    "Button",
                    "Snack",
                    "Fashion",
                    "Safety",
                    "Continuity",
                    "Singularity",
                    "Bandwidth",
                    "Telemetry",
                    "Magnet",
                    "Anomaly",
                    "Meme",
                ]
                noun_pool = [
                    "Director",
                    "Wrangler",
                    "Curator",
                    "Analyst",
                    "Mechanic",
                    "Coach",
                    "Consultant",
                    "Archivist",
                    "Custodian",
                    "Liaison",
                    "Specialist",
                    "Coordinator",
                    "Harmonizer",
                    "Designer",
                    "Operator",
                    "Technician",
                    "Sommelier",
                    "Supervisor",
                    "Navigator",
                    "Choreographer",
                ]
                entries: List[str] = []
                rng = random.Random(424242)
                for _ in range(320):
                    length = rng.choice((2, 3, 4))
                    words = [rng.choice(adjective_pool)]
                    while len(words) < length - 1:
                        words.append(rng.choice(middle_pool))
                    words.append(rng.choice(noun_pool))
                    entries.append(" ".join(words))
                entries.extend(
                    [
                        "Intergalactic Dad-Joke Distributor",
                        "Certified Meteor Petter",
                        "Fourth Wall Custodian",
                        "Time Loop Liaison",
                        "Chief Thing Officer",
                    ]
                )
                return entries
    
        def _ensure_layout_widths(self) -> None:
            if self._collapsed_width is not None and self._expanded_width is not None:
                return
            self._capture_initial_layout_widths()

        def _capture_initial_layout_widths(self) -> None:
            if self._content_frame is None:
                return
            self.update_idletasks()
            current_width = self.winfo_width()
            if current_width > 0:
                self._collapsed_width = current_width

            log_width = 0
            if self._log_box is not None:
                text_width = self._log_box.winfo_reqwidth()
                if self._log_scrollbar is not None:
                    text_width += self._log_scrollbar.winfo_reqwidth()
                log_width = max(log_width, text_width)
            if self._log_frame is not None:
                log_width = max(log_width, self._log_frame.winfo_reqwidth())

            if log_width <= 0:
                log_width = 260
            else:
                log_width += 24

            if self._collapsed_width is not None:
                candidate = self._collapsed_width + log_width
                if self._expanded_width is None or candidate > self._expanded_width:
                    self._expanded_width = candidate

        def _adjust_window_width(self, terminal_visible: bool) -> None:
            self._ensure_layout_widths()

            target_width: Optional[int]
            if terminal_visible:
                target_width = self._expanded_width
            else:
                target_width = self._collapsed_width

            if target_width is None:
                return

            self.update_idletasks()
            current_width = self.winfo_width()
            current_height = self.winfo_height()
            if current_height <= 0 and self._content_frame is not None:
                current_height = max(1, self._content_frame.winfo_reqheight())
            if abs(current_width - target_width) < 2:
                return

            x_pos = self.winfo_x()
            y_pos = self.winfo_y()
            self.geometry(f"{target_width}x{current_height}+{x_pos}+{y_pos}")
            self.update_idletasks()
            actual_width = self.winfo_width()
            if terminal_visible:
                if actual_width > target_width:
                    self._expanded_width = actual_width
            else:
                if self._collapsed_width is None or actual_width > self._collapsed_width:
                    self._collapsed_width = actual_width

        def _ensure_music_player(self) -> Optional[PixelShadowsPlayer]:
            if self.music_player is not None:
                return self.music_player
            try:
                self.music_player = PixelShadowsPlayer(self._music_search_roots)
            except Exception as exc:  # noqa: BLE001
                logging.debug("Unable to initialize music player: %s", exc)
                self.music_player = None
            return self.music_player

        def _on_disable_materials_toggle(self) -> None:
            disabled = self._disable_materials_var.get()
            if disabled and self._enable_sprite_materials_var.get():
                self._enable_sprite_materials_var.set(False)
            sprite_check = getattr(self, "_sprite_materials_check", None)
            if sprite_check is not None and sprite_check.winfo_exists():
                try:
                    if disabled:
                        sprite_check.state(["disabled"])
                    else:
                        sprite_check.state(["!disabled"])
                except Exception:
                    pass
                if sprite_check in self._config_control_states:
                    self._config_control_states[sprite_check] = "disabled" if disabled else "normal"

        def _capture_config_controls(self) -> None:
            interactive_types = (
                ttk.Entry,
                tk.Entry,
                ttk.Combobox,
                ttk.Button,
                ttk.Checkbutton,
                ttk.Radiobutton,
                ttk.Spinbox,
                ttk.Scale,
                ttk.Menubutton,
            )

            controls: List[tk.Widget] = []
            states: Dict[tk.Widget, str] = {}
            seen: Set[tk.Widget] = set()

            def walk(widget: tk.Widget) -> None:
                for child in widget.winfo_children():
                    walk(child)
                if isinstance(widget, interactive_types) and not getattr(widget, "_allow_during_run", False):
                    if widget in seen:
                        return
                    seen.add(widget)
                    try:
                        original_state = widget.cget("state")
                    except tk.TclError:
                        original_state = "normal"
                    controls.append(widget)
                    states[widget] = original_state

            walk(self)
            self._config_controls = controls
            self._config_control_states = states

        def _set_config_state(self, enabled: bool) -> None:
            if not self._config_controls:
                return
            for widget in list(self._config_controls):
                if not widget.winfo_exists():
                    continue
                original_state = self._config_control_states.get(widget, "normal")
                target_state = original_state if enabled else "disabled"
                try:
                    widget.configure(state=target_state)
                except tk.TclError:
                    try:
                        if enabled:
                            widget.state(["!disabled"])
                        else:
                            widget.state(["disabled"])
                    except Exception:
                        pass
            if enabled:
                self._on_disable_materials_toggle()

        def _poll_log_queue(self) -> None:
            while True:
                try:
                    line = self._log_queue.get_nowait()
                except queue.Empty:
                    break

                if line == "__ABORTED__":
                    self._status_var.set("Upscale aborted by user.")
                    self._progress_processed = 0
                    self._progress_total = 0
                    continue

                if line.startswith("__RETURN_CODE__"):
                    code = int(line.split("__RETURN_CODE__")[1])
                    if self._abort_requested:
                        self._status_var.set("Upscale aborted by user.")
                    elif code == 0:
                        self._status_var.set("Upscale finished. Enjoy the upgrade!")
                    else:
                        self._status_var.set(f"Upscale failed with exit code {code}.")
                    self._set_running(False)
                    self._worker = None
                    self._abort_requested = False
                    self._progress_processed = 0
                    self._progress_total = 0
                    continue

                progress_match = re.search(r"(\d+)\s+of\s+(\d+)\s+images processed", line)
                if not progress_match:
                    progress_match = re.search(r"Processed\s+(\d+)\s+of\s+(\d+)\s+images", line)
                if progress_match:
                    processed = int(progress_match.group(1))
                    total = int(progress_match.group(2))
                    self._progress_processed = processed
                    self._progress_total = total
                    if total > 0:
                        percent = max(0, min(100, int(round((processed / total) * 100))))
                        self._status_var.set(
                            f"Upscaling in progress... {percent}% ({processed}/{total})"
                        )
                    else:
                        self._status_var.set("Upscaling in progress...")

                self._append_log(line)

            self.after(100, self._poll_log_queue)

        def _append_log(self, line: str) -> None:
            if not self._log_box:
                return
            self._log_box.configure(state="normal")
            self._log_box.insert("end", line + "\n")
            self._log_box.see("end")
            self._log_box.configure(state="disabled")

        def _on_scale_change(self, value: str) -> None:
            try:
                numeric = int(round(float(value)))
            except (TypeError, ValueError):
                return
            self._scale_display_var.set(f"{numeric}x")

        def _on_music_toggle(self) -> None:
            if self._music_enabled_var.get():
                self._music_error_reported = False
                self._music_has_played = False
                player = self._ensure_music_player()
                if player is None:
                    self._status_var.set("Install pygame to enable the Pixel Shadows soundtrack.")
                    self._music_error_reported = True
                    return
                try:
                    player.start()
                except Exception as exc:  # noqa: BLE001
                    logging.debug("Unable to start soundtrack: %s", exc)
                    self._status_var.set("Unable to start soundtrack.")
                    self._music_error_reported = True
                    return
                self.after(1500, self._poll_music_status)
            else:
                if self.music_player is not None:
                    self.music_player.stop()
                self._music_error_reported = False
                self._music_has_played = False
                self.after(0, self._poll_music_status)

        def _detect_gpu_choices(self) -> list[str]:
            detected: List[str] = []
            torch_mod = globals().get("torch")
            if torch_mod is not None:
                try:
                    cuda_api = getattr(torch_mod, "cuda", None)
                    device_count_fn = getattr(cuda_api, "device_count", None) if cuda_api else None
                    device_count = int(device_count_fn()) if callable(device_count_fn) else 0
                except Exception:
                    device_count = 0
                if device_count > 0:
                    detected.extend(str(index) for index in range(device_count))
            env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
            if env_value:
                for token in env_value.split(","):
                    candidate = token.strip()
                    if candidate.isdigit() and candidate not in detected:
                        detected.append(candidate)
            if not detected:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                        capture_output=True,
                        check=True,
                        text=True,
                    )
                except Exception:
                    result = None
                if result and result.stdout:
                    for line in result.stdout.splitlines():
                        candidate = line.strip()
                        if candidate.isdigit() and candidate not in detected:
                            detected.append(candidate)
            if not detected:
                return ["auto"]
            detected.sort(key=int)
            return ["auto"] + detected

        def _setup_drag_and_drop(self) -> None:
            widgets = self._iter_drop_widgets()
            if not widgets:
                return

            package_loaded = False
            for package in ("tkdnd2.9", "tkdnd2.8", "tkdnd"):
                try:
                    self.tk.call("package", "require", package)
                    package_loaded = True
                    logging.debug("TkDND package '%s' loaded for drag-and-drop.", package)
                    break
                except tk.TclError:
                    continue

            if package_loaded:
                for widget in widgets:
                    try:
                        self.tk.call("tkdnd::drop_target", "register", widget._w, "DND_Files")
                        widget.bind("<<DragEnter>>", self._on_tkdnd_drag_enter, add="+")
                        widget.bind("<<DragPosition>>", self._on_tkdnd_drag_enter, add="+")
                        widget.bind("<<DragLeave>>", self._on_tkdnd_drag_leave, add="+")
                        widget.bind("<<Drop>>", self._on_tkdnd_drop, add="+")
                    except tk.TclError as exc:
                        logging.debug("Unable to register drag-and-drop for %s: %s", widget, exc)
                return

            if windnd is not None and sys.platform.startswith("win"):
                try:
                    windnd.hook_dropfiles(self, self._on_windnd_drop, force_unicode=True)
                    logging.debug("windnd drag-and-drop enabled.")
                    return
                except Exception as exc:  # noqa: BLE001
                    logging.debug("windnd drag-and-drop failed: %s", exc)

            logging.debug(
                "Drag-and-drop requires 'tkinterdnd2' or 'windnd'. Install one of these packages to enable it."
            )

        def _iter_drop_widgets(self) -> List[tk.Widget]:
            widgets: List[tk.Widget] = []

            def collect(widget: tk.Widget) -> None:
                widgets.append(widget)
                for child in widget.winfo_children():
                    collect(child)

            collect(self)
            return widgets

        def _on_tkdnd_drag_enter(self, event: tk.Event) -> str:
            return getattr(event, "action", "copy") or "copy"

        def _on_tkdnd_drag_leave(self, _event: tk.Event) -> str:
            return "copy"

        def _on_tkdnd_drop(self, event: tk.Event) -> str:
            data = getattr(event, "data", "")
            paths = self._parse_drop_data(data)
            self._handle_dropped_paths(paths)
            return getattr(event, "action", "copy") or "copy"

        def _parse_drop_data(self, data: str) -> list[str]:
            if not data:
                return []
            matches = re.findall(r"{([^}]*)}", data)
            tokens: List[str]
            if matches:
                tokens = matches
            else:
                try:
                    tokens = shlex.split(data)
                except ValueError:
                    tokens = data.split()
            cleaned: List[str] = []
            for token in tokens:
                candidate = token.strip().strip("{}")
                if candidate.startswith("file:///"):
                    candidate = candidate[8:]
                    if sys.platform.startswith("win"):
                        candidate = candidate.lstrip("/")
                candidate = candidate.replace("\\\\", "\\")
                if candidate:
                    cleaned.append(candidate)
            return cleaned

        def _handle_dropped_paths(self, paths: Sequence[str]) -> None:
            if not paths:
                return
            logging.debug("Processing dropped paths: %s", paths)
            normalized: List[Path] = []
            for raw in paths:
                cleaned = raw.strip().strip("{}\"'")
                if not cleaned:
                    continue
                candidate = Path(cleaned)
                normalized.append(candidate)
            if not normalized:
                return

            wad_path: Optional[Path] = next(
                (p for p in normalized if p.suffix.lower() in {".wad", ".pk3"} and p.exists()),
                None,
            )
            pth_path: Optional[Path] = next(
                (p for p in normalized if p.suffix.lower() == ".pth" and p.exists()),
                None,
            )

            if wad_path is not None:
                self._file_var.set(str(wad_path))
            else:
                fallback_wad = next(
                    (
                        p
                        for p in normalized
                        if p.exists() and p.suffix and p.suffix.lower() not in {".pth"}
                    ),
                    None,
                )
                if fallback_wad is not None:
                    self._file_var.set(str(fallback_wad))

            if pth_path is not None:
                self._apply_pth_selection(str(pth_path))

        def _on_windnd_drop(self, paths: Sequence[str]) -> None:
            decoded: List[str] = []
            for item in paths:
                if isinstance(item, bytes):
                    try:
                        decoded.append(item.decode("utf-8"))
                    except UnicodeDecodeError:
                        decoded.append(item.decode("mbcs", errors="ignore"))
                else:
                    decoded.append(str(item))
            self._handle_dropped_paths(decoded)


        def _apply_pth_selection(self, path: str) -> None:
            self._realesrgan_pth_var.set(path)
            if self._pth_combo is None:
                return
            try:
                current_values = list(self._pth_combo.cget("values"))
            except Exception:
                current_values = []
            if path not in current_values:
                current_values.append(path)
                self._pth_combo.config(values=current_values)

        def _get_eth_qr_photo(self) -> Optional[ImageTk.PhotoImage]:
            if self._qr_photo is None:
                image = _generate_eth_qr_image(ETH_DONATION_ADDRESS)
                if image is None:
                    logging.debug("ETH QR code unavailable; skipping donation QR rendering.")
                    return None
                width, height = image.size
                display_size = 128
                if width > 0 and height > 0:
                    resampling_namespace = getattr(Image, "Resampling", Image)
                    lanczos = getattr(resampling_namespace, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
                    nearest = getattr(resampling_namespace, "NEAREST", getattr(Image, "NEAREST", Image.NEAREST))
                    scale = display_size / float(max(width, height))
                    if not math.isclose(scale, 1.0, rel_tol=1e-6):
                        resample_mode = nearest if scale > 1.0 else lanczos
                        target_size = (
                            max(1, int(round(width * scale))),
                            max(1, int(round(height * scale))),
                        )
                        if target_size != (width, height):
                            try:
                                image = image.resize(target_size, resample_mode)
                                width, height = image.size
                            except Exception as exc:
                                logging.debug("Unable to scale ETH QR code image: %s", exc)
                    if (width, height) != (display_size, display_size):
                        try:
                            canvas_mode = "RGBA" if image.mode == "RGBA" else "RGB"
                            background = (255, 255, 255, 0) if canvas_mode == "RGBA" else "white"
                            canvas = Image.new(canvas_mode, (display_size, display_size), background)
                            offset = (
                                max(0, (display_size - width) // 2),
                                max(0, (display_size - height) // 2),
                            )
                            canvas.paste(image, offset)
                            image = canvas
                        except Exception as exc:
                            logging.debug("Unable to pad ETH QR code image: %s", exc)
                try:
                    self._qr_photo = ImageTk.PhotoImage(image)
                except Exception as exc:
                    logging.debug("Unable to convert ETH QR image to PhotoImage: %s", exc)
                    self._qr_photo = None
            return self._qr_photo

        def _update_skip_categories_selection(self) -> None:
            selected = [name for name, var in self._skip_category_vars.items() if var.get()]
            if selected:
                selected_sorted = sorted(selected)
                self._skip_categories_var.set(",".join(selected_sorted))
                pretty = ", ".join(name.upper() if len(name) <= 2 else name.title() for name in selected_sorted)
                self._skip_categories_display.set(pretty)
            else:
                self._skip_categories_var.set("")
                self._skip_categories_display.set("(none)")

        def _on_close(self) -> None:
            self._save_config_to_disk()
            if self.music_player is not None:
                self.music_player.stop()
            self.destroy()

        def _poll_music_status(self) -> None:
            if not self._music_enabled_var.get():
                if not self._music_error_reported:
                    self._status_var.set("Background music disabled.")
                    self._music_error_reported = True
                return
            if self._music_error_reported:
                return
            if self.music_player is None:
                player = self._ensure_music_player()
                if player is None:
                    self._status_var.set("Install pygame to enable the Pixel Shadows soundtrack.")
                    self._music_error_reported = True
                    return
            else:
                player = self.music_player

            if player.error:
                self._status_var.set(player.error)
                self._music_error_reported = True
                return
            if player.is_running():
                self._music_has_played = True
                self.after(1500, self._poll_music_status)
                return
            if self._music_has_played:
                self._status_var.set("Pixel Shadows ambience ready.")
                self._music_error_reported = True
                return
            self.after(500, self._poll_music_status)

        def _browse_output_dir(self) -> None:
            selected = filedialog.askdirectory(title="Select output directory")
            if selected:
                self._output_dir_var.set(selected)

        def _browse_realesrgan_bin(self) -> None:
            selected = filedialog.askopenfilename(
                title="Select realesrgan-ncnn-vulkan executable",
                filetypes=[("Executables", "*.exe;*.bin;*"), ("All files", "*.*")],
            )
            if selected:
                self._realesrgan_bin_var.set(selected)

        def _browse_realesrgan_pth(self) -> None:
            selected = filedialog.askopenfilename(
                title="Select RealESRGAN .pth model",
                filetypes=[("PyTorch weights", "*.pth"), ("All files", "*.*")],
            )
            if selected:
                self._realesrgan_pth_var.set(selected)

        def _collect_csv(self, raw: str) -> list[str]:
            tokens = []
            for part in re.split(r"[,\n]+", raw):
                token = part.strip()
                if token:
                    tokens.append(token)
            return tokens

        def _build_cli_args(self, input_path: Path, scale_value: int) -> Optional[list[str]]:
            args_list: list[str] = [str(input_path), "--scale", str(scale_value)]

            output_dir = self._output_dir_var.get().strip()
            if output_dir:
                args_list.extend(["--output-dir", output_dir])

            mode = self._output_mode_var.get()
            if mode not in {"diff", "full"}:
                messagebox.showerror("DOOM Upscaler", "Select a valid output mode (diff or full).")
                return None
            if mode == "full":
                args_list.extend(["--output-mode", "full"])

            if self._enable_sprite_materials_var.get():
                args_list.append("--enable-sprite-materials")
            if self._disable_materials_var.get():
                args_list.append("--disable-materials")
            if self._keep_temp_var.get():
                args_list.append("--keep-temp")
            if self._dry_run_var.get():
                args_list.append("--dry-run")
            if self._verbose_var.get():
                args_list.append("--verbose")
            if self._enable_post_sharpen_var.get():
                args_list.append("--enable-post-sharpen")

            detail_scale_str = self._detail_scale_var.get().strip()
            if detail_scale_str:
                try:
                    detail_scale_val = int(detail_scale_str)
                    if detail_scale_val <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("DOOM Upscaler", "Detail scale must be a positive integer.")
                    return None
                args_list.extend(["--detail-scale", str(detail_scale_val)])

            realesrgan_pth = self._realesrgan_pth_var.get().strip()
            if realesrgan_pth:
                args_list.extend(["--realesrgan-pth", realesrgan_pth])

            realesrgan_bin = self._realesrgan_bin_var.get().strip()
            if realesrgan_bin:
                args_list.extend(["--realesrgan-bin", realesrgan_bin])

            realesrgan_gpu = self._realesrgan_gpu_var.get().strip()
            if realesrgan_gpu:
                args_list.extend(["--realesrgan-gpu", realesrgan_gpu])

            tile_str = self._realesrgan_tile_var.get().strip()
            if tile_str:
                try:
                    tile_val = int(tile_str)
                    if tile_val < 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("DOOM Upscaler", "Tile size must be zero or a positive integer.")
                    return None
                args_list.extend(["--realesrgan-tile", str(tile_val)])

            tile_pad_str = self._realesrgan_tile_pad_var.get().strip()
            if tile_pad_str:
                try:
                    tile_pad_val = int(tile_pad_str)
                    if tile_pad_val < 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("DOOM Upscaler", "Tile padding must be zero or a positive integer.")
                    return None
                args_list.extend(["--realesrgan-tile-pad", str(tile_pad_val)])

            esrgan_strength_str = self._esrgan_strength_var.get().strip()
            if esrgan_strength_str:
                try:
                    esrgan_strength_val = float(esrgan_strength_str)
                except ValueError:
                    messagebox.showerror("DOOM Upscaler", "ESRGAN strength must be a number.")
                    return None
                args_list.extend(["--esrgan-strength", str(esrgan_strength_val)])

            esrgan_detail_limit_str = self._esrgan_detail_limit_var.get().strip()
            if esrgan_detail_limit_str:
                try:
                    esrgan_detail_limit_val = float(esrgan_detail_limit_str)
                except ValueError:
                    messagebox.showerror("DOOM Upscaler", "ESRGAN detail limit must be a number.")
                    return None
                args_list.extend(["--esrgan-detail-limit", str(esrgan_detail_limit_val)])

            max_pixels_str = self._max_pixels_var.get().strip()
            if max_pixels_str:
                try:
                    max_pixels_val = int(max_pixels_str)
                    if max_pixels_val < 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("DOOM Upscaler", "Max pixels must be zero or a positive integer.")
                    return None
                args_list.extend(["--max-pixels", str(max_pixels_val)])

            texture_exts = self._texture_exts_var.get().strip()
            if texture_exts:
                args_list.extend(["--texture-extensions", texture_exts])

            character_keywords = self._collect_csv(self._character_keywords_var.get())
            if character_keywords:
                args_list.extend(["--character-keywords", ",".join(character_keywords)])

            skip_categories = self._collect_csv(self._skip_categories_var.get())
            if skip_categories:
                args_list.extend(["--skip-types", ",".join(skip_categories)])

            return args_list

        def _start_window_move(self, event: tk.Event) -> None:
            self._is_dragging = True
            self._drag_offset_x = event.x
            self._drag_offset_y = event.y

        def _on_window_move(self, event: tk.Event) -> None:
            if not self._is_dragging:
                return
            x = event.x_root - self._drag_offset_x
            y = event.y_root - self._drag_offset_y
            self.geometry(f"+{x}+{y}")

        def _stop_window_move(self, _event: tk.Event) -> None:
            self._is_dragging = False

        def _minimize_window(self) -> None:
            self.overrideredirect(False)
            self.iconify()

        def _on_map(self, _event: tk.Event) -> None:
            self._focus_main_window()
            if self.state() == "normal":
                self.overrideredirect(True)

        def _on_unmap(self, _event: tk.Event) -> None:
            if self.state() == "iconic":
                self.after(20, lambda: self.overrideredirect(False))

        def _focus_main_window(self) -> None:
            if self._initial_focus_applied:
                return
            try:
                self.deiconify()
                self.lift()
                self.focus_force()
                self.attributes("-topmost", True)
            except Exception as exc:
                logging.debug("Unable to focus main window: %s", exc)
                self.after(150, self._focus_main_window)
                return
            self._initial_focus_applied = True
            self.after(200, self._clear_temp_topmost)

        def _clear_temp_topmost(self) -> None:
            try:
                self.attributes("-topmost", False)
            except Exception as exc:
                logging.debug("Unable to clear temporary topmost flag: %s", exc)
            else:
                self.after(50, lambda: self.focus_force())

    exec_dir_init, resource_dir_init = _runtime_directories()
    splash_search_roots = (
        resource_dir_init,
        exec_dir_init,
        Path.cwd(),
    )
    splash_music_player: Optional[PixelShadowsPlayer] = None
    if _is_music_enabled_in_config():
        try:
            splash_music_player = PixelShadowsPlayer(splash_search_roots)
            splash_music_player.start()
        except Exception as exc:
            logging.debug("Unable to start soundtrack before splash: %s", exc)
            splash_music_player = None
    _show_splash_window(exec_dir_init, resource_dir_init)
    app = TextureUpscalerLauncher(music_player=splash_music_player)
    app.mainloop()

def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    unified_model = derive_realesrgan_model_name(getattr(args, "realesrgan_pth", None))
    args.realesrgan_model_world = unified_model
    args.realesrgan_model_ui = unified_model
    requested_detail_scale = args.detail_scale

    if args.input_path is None:
        launch_gui()
        return

    input_path = args.input_path.resolve()
    if not input_path.exists():
        logging.error("Source file does not exist: %s", input_path)
        sys.exit(1)

    try:
        archive_type = detect_archive_type(input_path)
    except ValueError as exc:
        logging.error(str(exc))
        sys.exit(1)

    lib_model_path, lib_notes = resolve_realesrgan_pth(args)
    lib_upscaler: Optional["LibRealESRGANUpscaler"] = None
    realesrgan_runner: Optional[RealEsrganRunner] = None
    lib_errors: List[str] = []

    if args.detail_scale is None:
        if requested_detail_scale is not None:
            args.detail_scale = requested_detail_scale
        else:
            args.detail_scale = DEFAULT_DETAIL_SCALE

    if lib_model_path is not None:
        if LibRealESRGANUpscaler is None:
            lib_errors.append(
                "RealESRGAN PyTorch backend is unavailable. Install torch, torchvision, torchaudio, realesrgan, and spandrel."
            )
            if _LIB_IMPORT_ERROR:
                logging.debug("Lib backend import error: %s", _LIB_IMPORT_ERROR)
        else:
            try:
                init_scale = args.detail_scale or DEFAULT_DETAIL_SCALE
                half_precision_flag = True
                try:
                    import torch  # type: ignore

                    half_precision_flag = torch.cuda.is_available()
                except Exception:
                    half_precision_flag = False
                lib_upscaler = LibRealESRGANUpscaler(
                    scale=init_scale,
                    model_pth_path=str(lib_model_path),
                    half_precision=half_precision_flag,
                )
                device_name = getattr(lib_upscaler, "device_name", str(getattr(lib_upscaler, "device", "cuda")))
                logging.info(
                    "Using RealESRGAN models (lib backend, device=%s): model=%s",
                    device_name,
                    lib_model_path,
                )
                if getattr(lib_upscaler, "device", None) is not None and lib_upscaler.device.type == "cpu":  # type: ignore[attr-defined]
                    logging.info("RealESRGAN lib backend running on CPU with float32 precision for maximum quality.")
                lib_scale = getattr(lib_upscaler, "scale", init_scale)
                if lib_scale is not None and lib_scale != args.detail_scale:
                    args.detail_scale = lib_scale
                    logging.info("Detail scale set to %dx to match the loaded model.", lib_scale)
            except Exception as exc:  # noqa: BLE001
                lib_errors.append(f"Failed to initialize RealESRGAN lib backend: {exc}")
    else:
        for note in lib_notes:
            if note.startswith(("CLI", "REAL_ESRGAN_PTH")):
                lib_errors.append(note)
            else:
                logging.debug(note)
        lib_errors.append(
            "RealESRGAN .pth model not found. Provide a valid path via --realesrgan-pth or the REAL_ESRGAN_PTH environment variable."
        )

    if lib_upscaler is None:
        if lib_errors:
            for issue in lib_errors:
                logging.error(issue)
        else:
            logging.error("RealESRGAN PyTorch backend did not report a specific error before failing.")
        logging.error("RealESRGAN PyTorch backend is required but could not be initialized.")
        sys.exit(1)

    upscaler: UpscaleExecutor = RealEsrganUpscaler(
        runner=realesrgan_runner,
        lib_upscaler=lib_upscaler,
        post_sharpen=args.enable_post_sharpen,
        esrgan_strength=args.esrgan_strength,
        esrgan_detail_limit=args.esrgan_detail_limit,
    )

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_suffix = ".pk3" if args.output_mode == "diff" else input_path.suffix
    output_path = build_enhanced_name(output_dir / input_path.name, suffix=output_suffix)

    logging.info(
        "Processing %s (%s source, %s output mode)",
        input_path,
        archive_type.upper(),
        args.output_mode,
    )
    pix2vox_config: Optional[CharacterModelerConfig] = None
    character_output_root = args.character_output_dir.resolve()
    if args.pix2vox_command and args.pix2vox_weights:
        if not CHARACTER_MODELER_AVAILABLE:
            logging.error(
                "Pix2Vox+ integration requested but character_modeler.py could not be imported; skipping character modeling.",
            )
        else:
            source_dir_name = sanitize_pix2vox_name(input_path.stem)
            target_output = character_output_root / source_dir_name
            pix2vox_config = CharacterModelerConfig(
                command=args.pix2vox_command,
                weights=args.pix2vox_weights,
                output_dir=target_output,
                mesh_format=args.pix2vox_mesh_format,
                device=args.pix2vox_device,
                max_angles=max(1, int(args.pix2vox_max_angles)),
                fps=float(args.pix2vox_fps),
            )
    elif args.pix2vox_command or args.pix2vox_weights:
        logging.warning("Both --pix2vox-command and --pix2vox-weights are required to enable Pix2Vox+ modeling.")
    texture_exts = args.texture_extensions.split(",")
    for raw_keywords in getattr(args, "character_keywords", []):
        for token in raw_keywords.split(","):
            keyword = token.strip().lower()
            if keyword:
                EXTRA_CHARACTER_KEYWORDS.add(keyword)
    if EXTRA_CHARACTER_KEYWORDS:
        logging.info(
            "Using additional character keywords: %s",
            ", ".join(sorted(EXTRA_CHARACTER_KEYWORDS)),
        )
    requested_skips: set[str] = set()
    for raw_value in getattr(args, "skip_categories", []):
        for token in raw_value.split(","):
            category = token.strip().lower()
            if category:
                requested_skips.add(category)
    valid_categories = set(TEXTURE_CATEGORY_CHOICES)
    unknown_categories = requested_skips - valid_categories
    if unknown_categories:
        logging.warning(
            "Ignoring unknown texture categories in --skip-types: %s",
            ", ".join(sorted(unknown_categories)),
        )
        requested_skips -= unknown_categories
    skip_categories = requested_skips
    if skip_categories:
        logging.info(
            "Skipping texture categories: %s",
            ", ".join(sorted(skip_categories)),
        )
    target_scale = max(1, args.scale)
    detail_scale = max(1, args.detail_scale)
    max_pixels = max(0, args.max_pixels)
    if detail_scale < target_scale:
        scale_ratio = target_scale / detail_scale
        logging.info(
            "Model detail scale %dx is smaller than requested final scale %dx; pre-scaling inputs by %.3fx before AI inference.",
            detail_scale,
            target_scale,
            scale_ratio,
        )
    logging.info("Applying %dx AI upscale before bicubic downscale to %dx final size", detail_scale, target_scale)
    if max_pixels > 0:
        logging.info("Skipping textures larger than %d total pixels", max_pixels)

    try:
        if archive_type == "pk3":
            process_pk3(
                source=input_path,
                dest=output_path,
                upscaler=upscaler,
                detail_scale=detail_scale,
                target_scale=target_scale,
                max_pixels=max_pixels,
                texture_exts=texture_exts,
                skip_categories=skip_categories,
                generate_sprite_materials=args.enable_sprite_materials,
                disable_materials=args.disable_materials,
                keep_temp=args.keep_temp,
                dry_run=args.dry_run,
                output_mode=args.output_mode,
                character_model_config=pix2vox_config,
            )
        else:
            process_wad(
                source=input_path,
                dest=output_path,
                upscaler=upscaler,
                detail_scale=detail_scale,
                target_scale=target_scale,
                max_pixels=max_pixels,
                skip_categories=skip_categories,
                generate_sprite_materials=args.enable_sprite_materials,
                disable_materials=args.disable_materials,
                keep_temp=args.keep_temp,
                dry_run=args.dry_run,
                output_mode=args.output_mode,
                character_model_config=pix2vox_config,
            )
    except Exception as exc:  # noqa: BLE001 - top-level safety
        logging.error("Failed to enhance textures: %s", exc)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

