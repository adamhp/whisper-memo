# whisper-memo

Record a voice memo in your terminal, get the transcript on your clipboard.
Transcription runs entirely on your machine with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
No account, no API key, no audio leaves your computer.

![whisper-memo recording and transcribing a memo](https://raw.githubusercontent.com/adamhp/whisper-memo/main/docs/demo.gif)

## Install

With [uv](https://docs.astral.sh/uv/) (recommended) or [pipx](https://pipx.pypa.io/):

```sh
uv tool install whisper-memo          # CPU, works everywhere
uv tool install 'whisper-memo[cuda]'  # NVIDIA GPU on Linux or Windows
```

Replace `uv tool install` with `pipx install` if you use pipx.

The first run downloads the Whisper model from Hugging Face (about 1.6 GB for the default `large-v3-turbo`).
After that, everything works offline.

### Platform notes

- **Linux:** install PortAudio for the microphone (`sudo apt install libportaudio2`)
  and a clipboard tool (`xclip`, `xsel` or `wl-clipboard`).
- **macOS:** the terminal asks for microphone permission the first time. Transcription runs on the CPU;
  pick a smaller model (`-m small`) if it feels slow.
- **Windows:** works out of the box. Add `[cuda]` for an NVIDIA GPU.
- **WSL:** microphone access depends on WSLg audio support and may not work.

## Usage

```sh
whisper-memo                  # default model, English
whisper-memo -m small         # smaller, faster model
whisper-memo -l auto          # detect the spoken language
whisper-memo -d cpu           # force CPU even if a GPU is present
whisper-memo --no-clipboard   # print only
```

Press **Enter** to start recording and **Enter** again to stop. **Ctrl+C** quits.

Say **"new line"** while recording to split the transcript into bullet points:

> "Buy milk, new line, call mom, new line, book flights"

```
- Buy milk
- Call mom
- Book flights
```

| Option | Default | |
| --- | --- | --- |
| `-m`, `--model` | `large-v3-turbo` | Any [faster-whisper model](https://huggingface.co/Systran) name or a local path. Also read from `WHISPER_MODEL`. |
| `-d`, `--device` | `auto` | `auto`, `cuda` or `cpu`. `auto` uses the GPU when one is available. |
| `-l`, `--language` | `en` | Language code (`de`, `ja`, ...) or `auto`. |
| `--no-clipboard` | off | Don't copy transcripts to the clipboard. |

## Development

```sh
git clone https://github.com/adamhp/whisper-memo
cd whisper-memo
uv run whisper-memo                    # run from source
uv run --group dev pytest              # tests
uv run --group dev ruff check          # lint
uv run --group dev ruff format         # format
```

### Releasing

1. Bump `version` in `pyproject.toml` and move the `Unreleased` notes in `CHANGELOG.md` under the new version.
2. Commit, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The release workflow builds the package, publishes it to PyPI and creates a GitHub release.

## License

[MIT](LICENSE)
