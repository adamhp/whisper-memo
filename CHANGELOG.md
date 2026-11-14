# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-14

### Added

- `whisper-memo` command: press Enter to record, Enter to stop; the transcript is printed and copied to the clipboard.
- Live recording display with a timer and a mic-level waveform.
- Say "new line" while recording to turn the transcript into bullet points.
- `--model`, `--device`, `--language` and `--no-clipboard` options.
- Optional `cuda` extra that installs the NVIDIA libraries for GPU transcription.
