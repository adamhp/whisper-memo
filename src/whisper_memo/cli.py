"""Record voice memos from the default mic and transcribe them locally with faster-whisper.

Enter starts recording, Enter stops it. The transcript is printed and copied to the clipboard.
Ctrl+C quits.
"""

import argparse
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from whisper_memo import __version__

SAMPLE_RATE = 16000  # Whisper's native rate
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
STYLE_PROMPT = "Hello, welcome to my voice memo. Here are my thoughts, written in complete sentences."

console = Console()

BARS = " ▁▂▃▄▅▆▇█"
WAVE_WIDTH = 40


def preload_cuda_libs() -> None:
    """Make the cuBLAS/cuDNN libraries from the optional nvidia-* wheels visible to ctranslate2."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    roots = [Path(r) for r in (spec.submodule_search_locations if spec else [])]
    pkgs = ("cuda_nvrtc", "cublas", "cudnn")

    if sys.platform == "win32":
        # On Windows the wheels ship DLLs in nvidia/<pkg>/bin.
        for dll_dir in (root / pkg / "bin" for root in roots for pkg in pkgs):
            if dll_dir.is_dir():
                os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
    elif sys.platform.startswith("linux"):
        # On Linux the wheels put .so files in nvidia/<pkg>/lib. LD_LIBRARY_PATH is read only at
        # process start, so preload them globally; ctranslate2's later dlopen then finds them.
        import ctypes

        libs = [so for root in roots for pkg in pkgs for so in sorted((root / pkg / "lib").glob("lib*.so*"))]
        # Retry passes resolve inter-library dependencies regardless of load order.
        for _ in range(3):
            failed = []
            for so in libs:
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    failed.append(so)
            if not failed or len(failed) == len(libs):
                break
            libs = failed


def load_model(name: str, device: str):
    preload_cuda_libs()
    import ctranslate2
    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device == "cuda":
        try:
            return WhisperModel(name, device="cuda", compute_type="float16"), "cuda"
        except Exception as e:
            console.print(f"[yellow]CUDA unavailable ({e}); falling back to CPU.[/]")
    return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"


def level_bar(rms: float) -> str:
    """Map a chunk's RMS to a bar glyph on a log scale (-60 dB silent, -10 dB full)."""
    db = 20 * np.log10(rms + 1e-9)
    frac = min(max((db + 60) / 50, 0.0), 1.0)
    return BARS[round(frac * (len(BARS) - 1))]


class AudioError(RuntimeError):
    """The microphone or the PortAudio library is unavailable."""


def record() -> np.ndarray:
    try:
        import sounddevice as sd
    except OSError as e:  # raised when the PortAudio shared library is missing
        raise AudioError(e) from e

    chunks = []
    levels = deque(" " * WAVE_WIDTH, maxlen=WAVE_WIDTH)
    stop = threading.Event()

    def callback(indata, *_):
        chunks.append(indata.copy())
        levels.append(level_bar(float(np.sqrt(np.mean(indata**2)))))

    def render() -> Text:
        secs = int(sum(len(c) for c in chunks) / SAMPLE_RATE)
        blink = int(time.monotonic() * 2) % 2 == 0
        return Text.assemble(
            ("● " if blink else "  ", "bold red"),
            ("REC ", "bold red"),
            (f"{secs // 60:02d}:{secs % 60:02d}  ", "bold"),
            ("".join(levels), "cyan"),
            ("  Enter to stop", "dim"),
        )

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)
    except sd.PortAudioError as e:
        raise AudioError(e) from e
    with stream:
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        with Live(get_renderable=render, console=console, refresh_per_second=15, transient=True):
            stop.wait()

    return np.concatenate(chunks).flatten() if chunks else np.zeros(0, dtype=np.float32)


# Spoken "new line" (Whisper may render it "New line.", "new-line,", "newline" etc.).
NEW_LINE = re.compile(r"\bnew[\s-]?line\b[\s.,!?;:]*", re.IGNORECASE)


def format_bullets(text: str) -> str:
    """If the speaker said "new line", turn the pieces between them into bullet points."""
    if not NEW_LINE.search(text):
        return text
    items = (p.strip(" ,;:") for p in NEW_LINE.split(text))
    return "\n".join(f"- {item[0].upper()}{item[1:]}" for item in items if item)


def copy_to_clipboard(text: str) -> bool:
    import pyperclip

    try:
        pyperclip.copy(text)
        return True
    except pyperclip.PyperclipException:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="whisper-memo",
        description="Record voice memos and transcribe them locally with Whisper.",
        epilog='Say "new line" while recording to split the transcript into bullet points.',
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model name or local path (default: %(default)s; env: WHISPER_MODEL)",
    )
    parser.add_argument("-d", "--device", choices=("auto", "cuda", "cpu"), default="auto", help="default: %(default)s")
    parser.add_argument(
        "-l", "--language", default="en", help='spoken language code, or "auto" to detect (default: %(default)s)'
    )
    parser.add_argument("--no-clipboard", action="store_true", help="don't copy transcripts to the clipboard")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    language = None if args.language == "auto" else args.language

    with console.status(f"Loading [bold]{args.model}[/] (first run downloads it)...", spinner="dots"):
        model, device = load_model(args.model, args.device)
    console.print(f"[green]✓[/] [bold]{args.model}[/] ready on [bold]{device}[/]")

    while True:
        console.input("\n[dim]Press[/] [bold]Enter[/] [dim]to record, Ctrl+C to quit[/] ")
        audio = record()
        duration = audio.size / SAMPLE_RATE
        if audio.size < SAMPLE_RATE // 2:
            console.print("[yellow]Too short, skipped.[/]")
            continue

        start = time.monotonic()
        with console.status(f"Transcribing {duration:.1f}s of audio...", spinner="dots"):
            segments, _ = model.transcribe(
                audio,
                language=language,
                vad_filter=True,
                beam_size=5,
                # Whisper mimics the prompt's style; a punctuated prompt yields punctuated output.
                # The prompt is English, so only use it when transcribing English.
                initial_prompt=STYLE_PROMPT if language == "en" else None,
            )
            # segments is lazy; joining it runs the transcription, so keep it inside the spinner.
            text = format_bullets(" ".join(s.text.strip() for s in segments).strip())
        if not text:
            console.print("[yellow]No speech detected.[/]")
            continue

        status = f"transcribed in {time.monotonic() - start:.1f}s"
        if not args.no_clipboard:
            if copy_to_clipboard(text):
                status += " · copied to clipboard"
            else:
                status += " · [yellow]clipboard unavailable[/]"
        console.print(
            Panel(
                Text(text),
                title=f"{duration:.1f}s memo",
                subtitle=status,
                title_align="left",
                subtitle_align="right",
                border_style="green",
                padding=(1, 2),
            )
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        console.print("\n[dim]Bye.[/]")
    except AudioError as e:
        console.print(f"[red]Audio error:[/] {e}")
        console.print("[dim]Check that a microphone is connected and PortAudio is installed.[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
