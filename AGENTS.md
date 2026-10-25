## Project Overview

Sendspin CLI is a synchronized audio player client for the [Sendspin Protocol](https://github.com/Sendspin/website/blob/main/src/spec.md). It connects to Sendspin servers via WebSocket, receives time-synchronized audio streams, and plays them back with precise timing to enable multi-room synchronized audio.

**Note**: If uncertain about how something in Sendspin is supposed to work, fetch and refer to the [protocol specification](https://github.com/Sendspin/website/blob/main/src/spec.md) for authoritative implementation details.

## Architecture

```
                            cli.py (main entry)
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
                    ▼                               ▼
              ┌──────────┐                    ┌──────────┐
              │   tui/   │                    │ daemon/  │
              └──────────┘                    └──────────┘
                    │                               │
        ┌───────────┼───────────┐                   │
        │           │           │                   │
        ▼           ▼           ▼                   ▼
┌────────────┐ ┌─────────┐ ┌─────────┐      ┌─────────────┐
│ keyboard   │ │   ui    │ │   app   │      │   daemon    │
│            │ │         │ │         │      │             │
│ - Keys     │ │ - TUI   │ │ - State │      │ - Headless  │
│ - Commands │ │ - Panel │ │ - Tasks │      │ - Signals   │
└────────────┘ └─────────┘ └─────────┘      └─────────────┘
        │           │           │                   │
        └───────────┼───────────┴───────────────────┘
                    │
        ┌───────────┼───────────────────┐
        │           │                   │
        ▼           ▼                   ▼
┌─────────────┐ ┌───────────────┐ ┌──────────────┐
│ audio.py    │ │ discovery.py  │ │ aiosendspin  │
│             │ │               │ │ (external)   │
│ - Playback  │ │ - mDNS        │ │              │
│ - Time sync │ │ - Servers     │ │ - WebSocket  │
└─────────────┘ └───────────────┘ └──────────────┘
```

## File Responsibilities

### `README.md`
Project documentation, installation instructions, usage guide (including command line arguments), configuration options.

### `sendspin/cli.py`
Main entry point and orchestrator. Responsibilities:
- **Argument parsing**: Commands (`serve`, `daemon`), flags (`--url`, `--name`, `--id`, `--audio-device`, `--output-delay-ms`)
- **Command routing**: Routes to appropriate mode (TUI app, daemon, or server)
- **Device enumeration**: Lists audio devices and servers via `audio-devices list` and `servers list` subcommands
- **Backward compatibility**: Handles deprecated `--headless` and `--static-delay-ms` flags with warnings

### `sendspin/tui/` (TUI Mode Package)
Interactive terminal UI mode with keyboard control.

#### `sendspin/tui/app.py`
TUI mode application logic. Responsibilities:
- **Service discovery**: mDNS discovery of Sendspin servers
- **Connection management**: `ConnectionManager` handles reconnection with exponential backoff
- **State management**: `AppState` dataclass mirrors server state (playback, metadata, volume)
- **Audio stream handling**: `AudioStreamHandler` bridges between client and `AudioPlayer`
- **Event callbacks**: Routes server messages to appropriate handlers (metadata, group updates, commands)
- **UI coordination**: Manages `SendspinUI` and keyboard input loop

#### `sendspin/tui/keyboard.py`
Keyboard input handling for interactive control. Responsibilities:
- **Key capture**: Uses `readchar` library for single-keypress detection
- **Command parsing**: `CommandHandler` class parses and executes commands
- **Media commands**: play, pause, stop, next, previous, shuffle, repeat modes, switch group
- **Volume control**: Group volume (`vol+`/`vol-`/`mute`) and player volume (`pvol+`/`pvol-`/`pmute`)
- **Delay adjustment**: Real-time output delay adjustment via `delay` command
- **Keyboard shortcuts**: Arrow keys (prev/next/volume), space (toggle), m (mute), g (switch group), q (quit)

#### `sendspin/tui/ui.py`
Rich-based terminal UI for visual feedback. Responsibilities:
- **Now Playing panel**: Track title, artist, album with playback shortcuts
- **Volume panel**: Group and player volume with mute indicators
- **Progress bar**: Real-time track progress with interpolation during playback
- **Status line**: Connection status and quit shortcut
- **Shortcut highlighting**: Visual feedback when shortcuts are pressed (0.15s highlight)
- **State management**: `UIState` dataclass for all display state

### `sendspin/daemon/` (Daemon Mode Package)
Headless operation mode for background services.

#### `sendspin/daemon/daemon.py`
Daemon mode for headless operation (via `sendspin daemon` or deprecated `--headless`). Responsibilities:
- **Headless operation**: Runs without UI or keyboard input
- **Service discovery**: Same mDNS discovery as TUI mode
- **Connection management**: Reuses `ConnectionManager` from `tui/app.py`
- **Audio playback**: Same synchronized audio playback
- **Event logging**: Prints events to stdout instead of UI
- **Signal handling**: SIGINT/SIGTERM for graceful shutdown

### `sendspin/audio_devices.py`
Audio device resolution and ALSA device listing. Responsibilities:
- **Device resolution**: Resolves `--audio-device` argument by index, name prefix, or raw ALSA device name (via `resolve_audio_device()`)
- **ALSA device listing**: Enumerates ALSA PCM devices via `aplay -L` for display in `sendspin audio-devices list`
- **ALSA device fallback**: Opens ALSA plugin devices (dmix, plug) not enumerated by PortAudio, using safe defaults if PortAudio can't query device info

### `sendspin/audio.py`
Time-synchronized audio playback engine. Responsibilities:
- **Audio output**: Uses `sounddevice` (PortAudio) for low-latency playback
- **Time synchronization**: DAC-to-loop time calibration for precise playback timing
- **Buffer management**: Queue-based buffering with gap/overlap handling
- **Sync correction**: Sample drop/insert for drift correction (Kalman-filtered)
- **Playback state machine**: INITIALIZING → WAITING_FOR_START → PLAYING → REANCHORING
- **Volume control**: Software volume with power curve for natural control

### `sendspin/__init__.py`
Package entry point, exports `main` from `cli.py`.

## Key Dependencies

- **aiosendspin**: Async Sendspin protocol client (WebSocket, time sync, messages)
- **sounddevice**: PortAudio wrapper for audio I/O
- **rich**: Terminal UI rendering
- **readchar**: Cross-platform keyboard input
- **zeroconf**: mDNS service discovery
- **numpy**: Audio sample processing

## Common Tasks

### Running the player
```bash
uv run sendspin                                # TUI mode with auto-discovery
uv run sendspin --url ws://host:port/sendspin  # TUI mode with direct connection
uv run sendspin daemon                         # Daemon mode (headless) with auto-discovery
uv run sendspin daemon --url ws://host:port/sendspin  # Daemon mode with direct connection
```

### Development commands
```bash
uv run ruff check --fix .    # Lint and auto-fix
uv run ruff format .         # Format
uv run mypy sendspin         # Type check
```

### Before pushing commits
Run pre-commit before pushing any commit to ensure it is valid:
```bash
uv run pre-commit run --all-files
```

### Adding a new keyboard shortcut
1. Add key handler in `tui/keyboard.py` `keyboard_loop()` function
2. Add UI highlight call: `ui.highlight_shortcut("shortcut_name")`
3. Execute command via `handler.execute("command")`
4. Update shortcut display in `tui/ui.py` `_build_now_playing_panel()` or relevant panel

### Adding a new media command
1. Check if `MediaCommand` enum in aiosendspin supports it
2. Add command alias in `CommandHandler.execute()` in `tui/keyboard.py`
3. Command is sent via `client.send_group_command(MediaCommand.XXX)`

### Adding a new configuration option (CLI + settings file)
Options should be configurable via both CLI flags and the settings file, with CLI taking precedence.

1. **Add to `Settings` dataclass** in `sendspin/settings.py`:
   - Add field with default value (e.g., `use_mpris: bool = True`)
   - Add to `to_dict()` return value
   - Add to `from_dict()` with `data.get("field_name", default)`

2. **Add property to `SettingsManager`** in `sendspin/settings.py`:
   - Add `@property` that returns `self._settings.field_name`
   - Optionally add to `update()` method if it should be changeable at runtime

3. **Add CLI argument** in `sendspin/cli.py` `parse_args()`:
   - Add to both `daemon_parser` and main `parser` if applicable to both modes
   - Use `action="store_true"` for boolean flags

4. **Apply settings default** in `_apply_settings_defaults()`:
   - Compute final value: `args.option = <cli_value> or settings.option`
   - For disable flags: `args.use_x = not args.disable_x and settings.use_x`

5. **Add to Args dataclasses**:
   - Add field to `DaemonArgs` in `daemon/daemon.py`
   - Add field to `AppArgs` in `tui/app.py`

6. **Pass to apps** in `cli.py`:
   - Add `option=args.option` when creating `DaemonArgs` and `AppArgs`

7. **Use in daemon/TUI**:
   - Access via `self._args.option`

8. **Document** in `README.md` (required for user-facing features):
   - Add to example config JSON
   - Add to settings table
   - Add usage section if the feature warrants explanation

**Important:** All new user-facing features (CLI options, settings, commands) must be documented in `README.md`.
