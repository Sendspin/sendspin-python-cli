"""Rich-based terminal UI for the Sendspin CLI."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Any, Self

from PIL.Image import Image as PILImage
from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.types import PlaybackStateType, RepeatMode, UndefinedField
from aiosendspin.models.visualizer import BeatTiming
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sendspin.discovery import DiscoveredServer
from sendspin.tui.artwork import render_artwork
from sendspin.tui.visualizer import (
    BeatState,
    PeakEvent,
    PeakState,
    VisualizerState,
    freq_to_display_column,
    render_beat_strip,
    render_freq_cursor_row,
    render_peak_strip,
    render_spectrum,
)
from sendspin.utils import create_task


class _RefreshableLayout:
    """A renderable that rebuilds on each render cycle."""

    def __init__(self, ui: SendspinUI) -> None:
        self._ui = ui

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Rebuild and yield the layout on each render."""
        yield self._ui._build_layout()  # noqa: SLF001


# Duration in seconds to highlight a pressed shortcut
SHORTCUT_HIGHLIGHT_DURATION = 0.15
REFRESH_COALESCE_DELAY = 1 / 30
PLAYBACK_REFRESH_INTERVAL = 0.25
HIGHLIGHT_REFRESH_INTERVAL = 0.05
VISUALIZER_REFRESH_INTERVAL = 1 / 60
RESIZE_POLL_INTERVAL = 0.25


class ColorMode(Enum):
    """User-selected color theme. Active only when a palette is available."""

    DARK = "dark"
    LIGHT = "light"

    @classmethod
    def parse(cls, value: str | None) -> ColorMode:
        """Coerce a stored string to a mode, defaulting to DARK."""
        if value == cls.LIGHT.value:
            return cls.LIGHT
        return cls.DARK


# `dim` deliberately omitted: terminals render it ~50% grey, which fails the
# ≥4.5:1 contrast the spec requires against the artwork backgrounds.
_STYLE_MODIFIERS: frozenset[str] = frozenset(
    {"bold", "italic", "underline", "reverse", "strike", "blink"}
)


@dataclass
class UIState:
    """Holds state for the UI display."""

    # Connection
    server_url: str | None = None
    connected: bool = False
    status_message: str = "Initializing..."
    group_name: str | None = None

    # Server selector
    show_server_selector: bool = False
    available_servers: list[DiscoveredServer] = field(default_factory=list)
    selected_server_index: int = 0

    # Playback
    playback_state: PlaybackStateType | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    track_progress_ms: int | None = None
    track_duration_ms: int | None = None
    progress_updated_at: float = 0.0  # time.monotonic() when progress was updated

    # Volume
    volume: int | None = None
    muted: bool = False
    player_volume: int = 100
    player_muted: bool = False
    use_external_volume: bool = False

    # Audio format
    audio_codec: str | None = None
    audio_sample_rate: int = 0
    audio_bit_depth: int = 0
    audio_channels: int = 0

    # Delay
    delay_ms: float = 0.0

    # Repeat / Shuffle
    repeat_mode: RepeatMode | None = None
    shuffle: bool | None = None

    # Visualizer
    visualizer_enabled: bool = False
    # Visualizer types the server negotiated for the current stream.
    visualizer_types: frozenset[str] = field(default_factory=frozenset)
    visualizer_state: VisualizerState = field(default_factory=VisualizerState)
    beat_state: BeatState = field(default_factory=BeatState)
    peak_state: PeakState = field(default_factory=PeakState)
    # Synced server clock provider used for the beat and peak timeline strips.
    server_now_us: Callable[[], int] | None = None

    # Artwork palette pushed by the server via color@v1.
    palette_primary: tuple[int, int, int] | None = None
    palette_accent: tuple[int, int, int] | None = None
    palette_on_dark: tuple[int, int, int] | None = None
    palette_on_light: tuple[int, int, int] | None = None
    palette_background_dark: tuple[int, int, int] | None = None
    palette_background_light: tuple[int, int, int] | None = None
    # Gates themed rendering once the five spec-functional fields are present.
    palette_available: bool = False
    color_mode: ColorMode = ColorMode.DARK

    # Album artwork: None when unsupported or not yet received.
    artwork_image: PILImage | None = None
    # Bumped on every artwork update, used to key the renderable cache and
    # the now_playing panel cache.
    artwork_generation: int = 0

    # Shortcut highlight
    highlighted_shortcut: str | None = None
    highlight_time: float = 0.0

    # Pairing: the derived dynamic PIN, shown while a pairing exchange is in
    # progress. None when not pairing.
    pairing_code: str | None = None


class SendspinUI:
    """Rich-based terminal UI for the Sendspin CLI."""

    def __init__(
        self,
        delay_ms: float,
        *,
        player_volume: int = 100,
        player_muted: bool = False,
        use_external_volume: bool = False,
        visualizer_enabled: bool = False,
        color_mode: ColorMode = ColorMode.DARK,
        on_color_mode_change: Callable[[ColorMode], None] | None = None,
    ) -> None:
        """Initialize the UI."""
        self._console = Console()
        # Hex backgrounds need truecolor or 256-color support to keep contrast.
        self._supports_palette = self._console.color_system in ("truecolor", "256")
        self._on_color_mode_change = on_color_mode_change
        self._state = UIState(
            delay_ms=delay_ms,
            volume=player_volume,
            player_volume=player_volume,
            player_muted=player_muted,
            use_external_volume=use_external_volume,
            visualizer_enabled=visualizer_enabled,
            color_mode=color_mode,
        )
        self._live: Live | None = None
        self._running = False
        self._panel_cache: dict[str, tuple[tuple[Any, ...], Panel]] = {}
        self._dirty = False
        self._batch_depth = 0
        self._refresh_event = asyncio.Event()
        self._refresh_task: asyncio.Task[None] | None = None
        self._last_console_size: tuple[int, int] | None = None

    @property
    def state(self) -> UIState:
        """Get the UI state for external updates."""
        return self._state

    def _format_time(self, ms: int | None) -> str:
        """Format milliseconds as MM:SS."""
        if ms is None:
            return "--:--"
        seconds = ms // 1000
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    def _is_highlighted(self, shortcut: str) -> bool:
        """Check if a shortcut should be highlighted."""
        if self._state.highlighted_shortcut != shortcut:
            return False
        elapsed = time.monotonic() - self._state.highlight_time
        return elapsed < SHORTCUT_HIGHLIGHT_DURATION

    def _has_active_highlight(self) -> bool:
        """Check if any shortcut highlight animation is still active."""
        if self._state.highlighted_shortcut is None:
            return False
        elapsed = time.monotonic() - self._state.highlight_time
        return elapsed < SHORTCUT_HIGHLIGHT_DURATION

    def _needs_playback_refresh(self) -> bool:
        """Check if the progress panel needs periodic refreshes."""
        return (
            self._state.playback_state == PlaybackStateType.PLAYING
            and self._state.progress_updated_at > 0
            and (self._state.track_duration_ms or 0) > 0
        )

    def _needs_visualizer_refresh(self) -> bool:
        """Check if the visualizer needs periodic refreshes for interpolation."""
        if not self._state.visualizer_enabled:
            return False
        return (
            self._state.visualizer_state.is_active
            or self._state.beat_state.is_active
            or self._state.peak_state.is_active
        )

    def _next_refresh_interval(self) -> float | None:
        """Return the next periodic refresh interval, if any."""
        intervals: list[float] = []
        if self._needs_playback_refresh():
            intervals.append(PLAYBACK_REFRESH_INTERVAL)
        if self._has_active_highlight():
            intervals.append(HIGHLIGHT_REFRESH_INTERVAL)
        if self._needs_visualizer_refresh():
            intervals.append(VISUALIZER_REFRESH_INTERVAL)
        return min(intervals) if intervals else None

    def _flush_refresh(self, *, force: bool = False) -> None:
        """Refresh the live display if the UI is dirty or animating."""
        if self._live is None or not (self._dirty or force):
            return
        self._update_console_size()
        self._dirty = False
        self._live.refresh()

    def _update_console_size(self) -> bool:
        """Track terminal dimensions and report when they change."""
        size = self._console.size
        current_size = (size.width, size.height)
        changed = self._last_console_size is not None and current_size != self._last_console_size
        self._last_console_size = current_size
        return changed

    async def _refresh_loop(self) -> None:
        """Coalesce dirty updates and drive the few animations the UI uses."""
        event = self._refresh_event

        while self._running:
            animation_interval = None if self._dirty else self._next_refresh_interval()
            interval = (
                REFRESH_COALESCE_DELAY
                if self._dirty
                else animation_interval or RESIZE_POLL_INTERVAL
            )

            try:
                await asyncio.wait_for(event.wait(), timeout=interval)
            except TimeoutError:
                event.clear()
                if self._update_console_size():
                    self._dirty = True
                self._flush_refresh(force=animation_interval is not None and not self._dirty)
            except asyncio.CancelledError:
                break
            else:
                event.clear()
                self._flush_refresh()

    def _shortcut_style(self, shortcut: str) -> str:
        """Get the style for a shortcut key."""
        raw = "bold yellow reverse" if self._is_highlighted(shortcut) else "bold cyan"
        return self._themed(raw)

    def _cached_panel(self, name: str, key: tuple[Any, ...], builder: Callable[[], Panel]) -> Panel:
        """Return cached panel if key matches, otherwise rebuild and cache."""
        entry = self._panel_cache.get(name)
        if entry is not None and entry[0] == key:
            return entry[1]
        panel = builder()
        self._panel_cache[name] = (key, panel)
        return panel

    def highlight_shortcut(self, shortcut: str) -> None:
        """Highlight a shortcut temporarily."""
        self._state.highlighted_shortcut = shortcut
        self._state.highlight_time = time.monotonic()
        self.refresh()

    def _build_now_playing_panel(self, *, expand: bool = False) -> Panel:
        """Build the now playing panel."""
        is_active = self._state.playback_state is not None or self._state.title is not None

        # Show prompt when nothing is playing
        if not is_active:
            content = Table.grid()
            content.add_column()
            content.add_row("")
            line1 = Text()
            line1.append("Press ", style=self._themed("dim"))
            line1.append("<space>", style=self._themed("bold cyan"))
            line1.append(" to start playing", style=self._themed("dim"))
            content.add_row(line1)
            line2 = Text()
            line2.append("Press ", style=self._themed("dim"))
            line2.append("g", style=self._themed("bold cyan"))
            line2.append(" to join an existing session", style=self._themed("dim"))
            content.add_row(line2)
            content.add_row("")
            content.add_row("")
            return self._make_panel(
                content, title="Now Playing", default_border="blue", expand=expand
            )

        # Info grid with label/value columns
        info = Table.grid(padding=(0, 1))
        info.add_column(style=self._themed("dim"), width=8)
        info.add_column()

        if self._state.title:
            info.add_row("Title:", Text(self._state.title, style=self._themed("bold white")))
            info.add_row(
                "Artist:", Text(self._state.artist or "Unknown artist", style=self._themed("cyan"))
            )
            info.add_row(
                "Album:", Text(self._state.album or "Unknown album", style=self._themed("dim"))
            )
        else:
            state_label = (
                self._state.playback_state.value.capitalize()
                if self._state.playback_state
                else "Active"
            )
            info.add_row("Status:", Text(state_label, style=self._themed("bold white")))
            info.add_row("", Text("No metadata available", style=self._themed("dim")))
            info.add_row("")

        # Metadata + shortcuts column (what the panel showed before this change)
        metadata_col = Table.grid()
        metadata_col.add_column()
        metadata_col.add_row(info)
        metadata_col.add_row("")  # spacing

        space_label = "pause" if self._state.playback_state == PlaybackStateType.PLAYING else "play"
        shortcuts = Text()
        shortcuts.append("←", style=self._shortcut_style("prev"))
        shortcuts.append(" prev  ", style=self._themed("dim"))
        shortcuts.append("<space>", style=self._shortcut_style("space"))
        shortcuts.append(f" {space_label}  ", style=self._themed("dim"))
        shortcuts.append("→", style=self._shortcut_style("next"))
        shortcuts.append(" next", style=self._themed("dim"))
        metadata_col.add_row(shortcuts)

        # Wrap with an artwork column on the left when artwork is available
        # and the layout is wide enough.
        artwork = render_artwork(
            self._state.artwork_image,
            self._state.artwork_generation,
            height_rows=5,
            width_cells=10,
        )
        narrow = self._console.width - 1 < 80
        if artwork is not None and not narrow:
            outer = Table.grid(padding=(0, 2))
            outer.add_column()
            outer.add_column()
            outer.add_row(artwork, metadata_col)
            content = outer
        else:
            content = metadata_col

        return self._make_panel(content, title="Now Playing", default_border="blue", expand=expand)

    _PALETTE_REQUIRED = (
        "palette_primary",
        "palette_on_dark",
        "palette_on_light",
        "palette_background_dark",
        "palette_background_light",
    )

    def _palette_active(self) -> bool:
        """True when palette is complete and the terminal can render it."""
        return self._state.palette_available and self._supports_palette

    def _themed(self, style: str) -> str:
        """Force palette-contrast text color, keep only Rich modifiers."""
        if not self._palette_active():
            return style
        target = "white" if self._state.color_mode == ColorMode.DARK else "black"
        mods = [t for t in style.split() if t in _STYLE_MODIFIERS]
        return " ".join([target, *mods])

    def _palette_bg_hex(self) -> str | None:
        """Background hex for the active mode, or None if palette inactive."""
        if not self._palette_active():
            return None
        if self._state.color_mode == ColorMode.DARK:
            bg = self._state.palette_background_dark
        else:
            bg = self._state.palette_background_light
        if bg is None:
            return None
        return f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}"

    def _palette_fg(self) -> str:
        """Spec-mandated text color paired with the active background."""
        return "black" if self._state.color_mode == ColorMode.LIGHT else "white"

    def _palette_border_style(self) -> str | None:
        """Border color: the on-color for the active mode."""
        if not self._palette_active():
            return None
        if self._state.color_mode == ColorMode.DARK:
            border = self._state.palette_on_dark
        else:
            border = self._state.palette_on_light
        if border is None:
            return None
        return f"#{border[0]:02x}{border[1]:02x}{border[2]:02x}"

    def _palette_panel_style(self) -> str | None:
        """Rich `<fg> on <bg>` style painting the panel with the album background."""
        bg = self._palette_bg_hex()
        if bg is None:
            return None
        return f"{self._palette_fg()} on {bg}"

    def update_palette(self, color: SessionUpdateColor | None) -> None:
        """Merge a color@v1 payload into state and invalidate cached panels."""
        # UndefinedField keeps current value, None clears, tuple sets.
        if color is None:
            updates: dict[str, tuple[int, int, int] | None] = dict.fromkeys(
                ("primary", "accent", "on_dark", "on_light", "background_dark", "background_light")
            )
        else:
            updates = {
                attr: getattr(color, attr)
                for attr in (
                    "primary",
                    "accent",
                    "on_dark",
                    "on_light",
                    "background_dark",
                    "background_light",
                )
                if not isinstance(getattr(color, attr), UndefinedField)
            }

        changed = False
        for attr, value in updates.items():
            state_attr = f"palette_{attr}"
            if getattr(self._state, state_attr) != value:
                setattr(self._state, state_attr, value)
                changed = True
        if not changed:
            return

        # Accent is best-effort per spec, so gate only on the five required fields.
        all_present = all(getattr(self._state, attr) is not None for attr in self._PALETTE_REQUIRED)
        self._state.palette_available = all_present
        self._panel_cache.clear()
        self.refresh()

    def reset_palette(self) -> None:
        """Clear palette state. Used when the active server disconnects."""
        self.update_palette(None)

    def cycle_color_mode(self) -> None:
        """Toggle DARK ↔ LIGHT and persist the preference."""
        new_mode = ColorMode.LIGHT if self._state.color_mode == ColorMode.DARK else ColorMode.DARK
        self._state.color_mode = new_mode
        if self._on_color_mode_change is not None:
            self._on_color_mode_change(new_mode)
        if self._palette_active():
            self._panel_cache.clear()
            self.refresh()

    def _make_panel(
        self,
        content: Any,
        *,
        title: str,
        default_border: str,
        expand: bool = False,
    ) -> Panel:
        """Build a Panel tinted with the current artwork palette when active."""
        border = self._palette_border_style() or default_border
        style = self._palette_panel_style()
        if style is None:
            return Panel(content, title=title, border_style=border, expand=expand)
        return Panel(content, title=title, border_style=border, style=style, expand=expand)

    def _build_progress_bar(self, *, expand: bool = False) -> Panel:
        """Build the progress bar panel."""
        progress_ms = self._state.track_progress_ms or 0
        duration_ms = self._state.track_duration_ms or 0

        # Interpolate progress if playing
        if (
            self._state.playback_state == PlaybackStateType.PLAYING
            and self._state.progress_updated_at > 0
            and duration_ms > 0
        ):
            elapsed_ms = (time.monotonic() - self._state.progress_updated_at) * 1000
            progress_ms += int(elapsed_ms)

        if duration_ms > 0:
            progress_ms = max(0, min(progress_ms, duration_ms))

        percentage = progress_ms / duration_ms * 100 if duration_ms > 0 else 0

        # Time text (fixed width)
        time_str = f"{self._format_time(progress_ms)} / {self._format_time(duration_ms)}"

        # Calculate bar width: terminal - panel borders (4) - time text - spacing
        bar_width = max(10, self._console.width - 4 - len(time_str) - 5)
        filled = int(bar_width * percentage / 100)
        empty = bar_width - filled

        bar = Text()
        bar.append("[", style=self._themed("dim"))
        bar.append("=" * filled, style=self._themed("green bold"))
        if filled < bar_width:
            bar.append(">", style=self._themed("green bold"))
            bar.append("-" * max(0, empty - 1), style=self._themed("dim"))
        bar.append("] ", style=self._themed("dim"))

        time_text_styled = Text()
        time_text_styled.append(self._format_time(progress_ms), style=self._themed("cyan"))
        time_text_styled.append(" / ", style=self._themed("dim"))
        time_text_styled.append(self._format_time(duration_ms), style=self._themed("cyan"))

        # Use grid to keep bar and time on same line
        content = Table.grid(expand=True, padding=0)
        content.add_column()
        content.add_column(justify="right", no_wrap=True)
        content.add_row(bar, time_text_styled)

        return self._make_panel(content, title="Progress", default_border="green", expand=expand)

    def _build_volume_panel(self, *, expand: bool = False) -> Panel:
        """Build the volume panel."""
        # Info grid with label/value columns
        info = Table.grid(padding=(0, 2))
        info.add_column()
        info.add_column()

        # Group volume
        vol = self._state.volume if self._state.volume is not None else 0
        vol_style = self._themed("red" if self._state.muted else "cyan")
        vol_text = f"{vol}%" + (" [MUTED]" if self._state.muted else "")
        info.add_row("Group:", Text(vol_text, style=vol_style))

        # Player volume
        pvol = self._state.player_volume
        pvol_style = self._themed("red" if self._state.player_muted else "cyan")
        pvol_text = f"{pvol}%" + (" [MUTED]" if self._state.player_muted else "")
        player_label = "External:" if self._state.use_external_volume else "Player:"
        info.add_row(player_label, Text(pvol_text, style=pvol_style))

        # Vertical container for info + shortcuts
        content = Table.grid()
        content.add_column()
        content.add_row(info)
        content.add_row("")  # Spacing

        # Player volume shortcuts
        player_sc = Text()
        player_sc.append("↑", style=self._shortcut_style("up"))
        player_sc.append("/", style=self._themed("dim"))
        player_sc.append("↓", style=self._shortcut_style("down"))
        player_sc.append(" player  ", style=self._themed("dim"))
        player_sc.append("m", style=self._shortcut_style("mute"))
        player_sc.append(" mute", style=self._themed("dim"))
        content.add_row(player_sc)

        # Group volume shortcuts
        group_sc = Text()
        group_sc.append("[", style=self._shortcut_style("group-down"))
        group_sc.append("/", style=self._themed("dim"))
        group_sc.append("]", style=self._shortcut_style("group-up"))
        group_sc.append(" group  ", style=self._themed("dim"))
        group_sc.append("M", style=self._shortcut_style("group-mute"))
        group_sc.append(" mute", style=self._themed("dim"))
        content.add_row(group_sc)

        return self._make_panel(content, title="Volume", default_border="magenta", expand=expand)

    def _build_connection_panel(self, *, expand: bool = False) -> Panel:
        """Build the connection status panel."""
        content = Table.grid(padding=(0, 1))
        content.add_column(style=self._themed("dim"), width=8)
        content.add_column()

        if self._state.connected and self._state.server_url:
            status = Text("Connected", style=self._themed("green bold"))
            url = Text(self._state.server_url, style=self._themed("cyan"))
        else:
            status = Text("Disconnected", style=self._themed("red bold"))
            url = Text(self._state.status_message, style=self._themed("yellow"))

        content.add_row("Status:", status)
        content.add_row("Server:", url)

        return self._make_panel(content, title="Connection", default_border="yellow", expand=expand)

    def _build_server_selector_panel(self) -> Panel:
        """Build the server selector panel."""
        content = Table.grid()
        content.add_column()

        if not self._state.available_servers:
            content.add_row("")
            content.add_row(Text("Searching for servers...", style=self._themed("dim")))
            content.add_row("")
        else:
            for i, server in enumerate(self._state.available_servers):
                is_selected = i == self._state.selected_server_index
                is_current = server.url == self._state.server_url

                line = Text()
                if is_selected:
                    line.append(" > ", style=self._themed("bold cyan"))
                else:
                    line.append("   ")

                # Server name
                name_style = "bold white" if is_selected else "white"
                line.append(server.name, style=name_style)

                # Current server indicator
                if is_current:
                    line.append(" (current)", style=self._themed("dim green"))

                content.add_row(line)

                # Show URL below name
                url_line = Text()
                url_line.append("   ")
                url_style = "cyan" if is_selected else "dim"
                url_line.append(f"   {server.host}:{server.port}", style=url_style)
                content.add_row(url_line)

        content.add_row("")

        # Shortcuts
        shortcuts = Text()
        shortcuts.append("↑", style=self._shortcut_style("selector-up"))
        shortcuts.append("/", style=self._themed("dim"))
        shortcuts.append("↓", style=self._shortcut_style("selector-down"))
        shortcuts.append(" navigate  ", style=self._themed("dim"))
        shortcuts.append("<enter>", style=self._shortcut_style("selector-enter"))
        shortcuts.append(" connect  ", style=self._themed("dim"))
        shortcuts.append("r", style=self._shortcut_style("selector-enter"))
        shortcuts.append(" refresh  ", style=self._themed("dim"))
        shortcuts.append("q", style=self._shortcut_style("selector-enter"))
        shortcuts.append(" back", style=self._themed("dim"))
        content.add_row(shortcuts)

        return self._make_panel(content, title="Select Server", default_border="cyan")

    def _build_pairing_panel(self) -> Panel:
        """Build the pairing panel: the derived dynamic PIN."""
        content = Table.grid()
        content.add_column(justify="center")

        content.add_row(Text("Pairing with server", style=self._themed("bold")))
        content.add_row("")

        if self._state.pairing_code:
            code = Text(self._state.pairing_code, style=self._themed("bold cyan"))
            content.add_row(code)
            content.add_row("")

        content.add_row(
            Text("Approve this client on the server to finish pairing.", style=self._themed("dim"))
        )

        return self._make_panel(content, title="Pairing Required", default_border="cyan")

    def _build_playback_panel(self, *, expand: bool = False, min_info_rows: int = 0) -> Panel:
        """Build the playback panel with repeat/shuffle status."""
        info = Table.grid(padding=(0, 2))
        info.add_column(style=self._themed("dim"), width=8)
        info.add_column()

        repeat = self._state.repeat_mode
        info.add_row(
            "Repeat:",
            Text(
                repeat.value if repeat is not None else "—",
                style=self._themed("cyan" if repeat else "dim"),
            ),
        )

        shuffle = self._state.shuffle
        if shuffle is not None:
            shuffle_text = Text("on" if shuffle else "off", style=self._themed("cyan"))
        else:
            shuffle_text = Text("—", style=self._themed("dim"))
        info.add_row("Shuffle:", shuffle_text)
        info_rows = 2

        content = Table.grid()
        content.add_column()
        content.add_row(info)
        for _ in range(max(0, min_info_rows - info_rows)):
            content.add_row("")
        content.add_row("")  # Spacing before shortcuts

        # Shortcuts
        shortcuts = Text()
        shortcuts.append("r", style=self._shortcut_style("repeat"))
        shortcuts.append(" repeat  ", style=self._themed("dim"))
        shortcuts.append("x", style=self._shortcut_style("shuffle"))
        shortcuts.append(" shuffle", style=self._themed("dim"))
        content.add_row(shortcuts)

        return self._make_panel(content, title="Playback", default_border="yellow", expand=expand)

    def _build_stream_quality_panel(self, *, expand: bool = False, min_info_rows: int = 0) -> Panel:
        """Build the stream quality panel."""
        info = Table.grid(padding=(0, 1))
        info.add_column(style=self._themed("dim"))
        info.add_column()

        if self._state.audio_sample_rate > 0:
            codec_label = (self._state.audio_codec or "PCM").upper()
            info.add_row("Codec:", Text(codec_label, style=self._themed("cyan")))
            rate_khz = self._state.audio_sample_rate / 1000
            info.add_row("Rate:", Text(f"{rate_khz:.1f}kHz", style=self._themed("cyan")))
            info.add_row(
                "Depth:", Text(f"{self._state.audio_bit_depth}bit", style=self._themed("cyan"))
            )
            ch_label = (
                "Stereo" if self._state.audio_channels == 2 else f"{self._state.audio_channels}ch"
            )
            info.add_row("Channels:", Text(ch_label, style=self._themed("cyan")))
        else:
            info.add_row("Codec:", Text("—", style=self._themed("dim")))
            info.add_row("Rate:", Text("—", style=self._themed("dim")))
            info.add_row("Depth:", Text("—", style=self._themed("dim")))
            info.add_row("Channels:", Text("—", style=self._themed("dim")))

        delay = self._state.delay_ms
        delay_str = f"+{delay:.0f}ms" if delay >= 0 else f"{delay:.0f}ms"
        info.add_row("Delay:", Text(delay_str, style=self._themed("cyan")))
        info_rows = 5

        content = Table.grid()
        content.add_column()
        content.add_row(info)
        for _ in range(max(0, min_info_rows - info_rows)):
            content.add_row("")
        content.add_row("")  # Spacing before shortcuts

        # Shortcuts
        shortcuts = Text()
        shortcuts.append(",", style=self._shortcut_style("delay-"))
        shortcuts.append("/", style=self._themed("dim"))
        shortcuts.append(".", style=self._shortcut_style("delay+"))
        shortcuts.append(" adjust delay", style=self._themed("dim"))
        content.add_row(shortcuts)

        return self._make_panel(content, title="Stream", default_border="yellow", expand=expand)

    def _build_server_panel(self, *, expand: bool = False, min_info_rows: int = 0) -> Panel:
        """Build the server panel."""
        info = Table.grid(padding=(0, 1))
        info.add_column(style=self._themed("dim"))
        info.add_column()

        if self._state.connected and self._state.server_url:
            parsed = urlparse(self._state.server_url)
            host = parsed.hostname or ""
            port = str(parsed.port) if parsed.port else ""
            path = parsed.path or "/"
            info.add_row("Status:", Text("Connected", style=self._themed("green bold")))
            info.add_row("Host:", Text(host, style=self._themed("cyan")))
            if port:
                info.add_row("Port:", Text(port, style=self._themed("cyan")))
            info.add_row("Path:", Text(path, style=self._themed("cyan")))
            if self._state.group_name:
                info.add_row("Group:", Text(self._state.group_name, style=self._themed("cyan")))
            info_rows = 3 + (1 if port else 0) + (1 if self._state.group_name else 0)
        else:
            info.add_row("Status:", Text("Disconnected", style=self._themed("red bold")))
            info.add_row("Host:", Text(self._state.status_message, style=self._themed("yellow")))
            info_rows = 2

        content = Table.grid()
        content.add_column()
        content.add_row(info)
        for _ in range(max(0, min_info_rows - info_rows)):
            content.add_row("")
        # Shortcuts
        shortcut_group = Text()
        shortcut_group.append("g", style=self._shortcut_style("switch"))
        shortcut_group.append(" change group", style=self._themed("dim"))
        content.add_row(shortcut_group)
        shortcut_server = Text()
        shortcut_server.append("s", style=self._shortcut_style("server"))
        shortcut_server.append(" change server", style=self._themed("dim"))
        content.add_row(shortcut_server)

        return self._make_panel(content, title="Server", default_border="yellow", expand=expand)

    def _build_visualizer_rows(self, height: int) -> list[Text]:
        """Build the visualizer as raw Text rows, totaling `height`.

        Stacks the peak and beat strips above the spectrum and the f_peak
        arrow, pitch arrow, and footer below it, dropping lowest-priority
        rows first on short terminals. The spectrum fills the rest.
        """
        state = self._state.visualizer_state
        state.step()
        magnitudes = state.get_spectrum()
        loudness = state.loudness
        peaks = state.get_peaks()
        beat_pulse = self._state.beat_state.pulse_intensity()

        # Two guaranteed-contrast colors per the color spec: the on-color
        # (on_dark/on_light, >=4.5:1 vs its background) and white/black text
        # (guaranteed vs background_dark/background_light). primary/accent carry
        # NO contrast guarantee, so they are never used against the painted bg.
        palette_on = self._palette_active()
        if not palette_on:
            palette_low = None
            palette_high = None
            on_color = "#ffffff"
            text_color = "#ffffff"
        elif self._state.color_mode == ColorMode.DARK:
            palette_low = self._state.palette_background_light
            palette_high = self._state.palette_on_dark
            text_color = "#ffffff"
            assert palette_high is not None
            on_color = f"#{palette_high[0]:02x}{palette_high[1]:02x}{palette_high[2]:02x}"
        else:
            palette_low = self._state.palette_background_dark
            palette_high = self._state.palette_on_light
            text_color = "#000000"
            assert palette_high is not None
            on_color = f"#{palette_high[0]:02x}{palette_high[1]:02x}{palette_high[2]:02x}"

        bar_width = max(10, self._console.width - 1)
        bg_color = self._palette_bg_hex()
        # Same background the spectrum paints, so the strips/cursor/footer match.
        row_bg = f"on {bg_color}" if bg_color else ""

        # Frequency-domain cursors: arrows onto the spectrum's freq axis plus a
        # footer naming each value. pitch uses the on-color, f_peak white/black —
        # both guaranteed against the background, and distinct from each other.
        f_peak_marker, pitch_marker, footer_parts, f_peak_col = self._build_freq_cursors(
            bar_width, on_color, text_color
        )
        vtypes = self._state.visualizer_types
        clock = self._state.server_now_us

        # Row budget, highest keep-priority first: peaks, beats (above the
        # spectrum), then the f_peak arrow, pitch arrow, and footer (below it).
        # Strips and f_peak need their type negotiated. A negotiated pitch keeps
        # its row reserved so the arrow appearing/vanishing with confidence
        # doesn't shift the layout; an unnegotiated pitch still shows on data.
        # On short terminals the lowest-priority rows drop first, spectrum >=1.
        has_pitch = self._state.visualizer_state.has_pitch
        show_pitch_row = "pitch" in vtypes or has_pitch
        candidates: list[str] = []
        if clock is not None and "peak" in vtypes:
            candidates.append("peak")
        if clock is not None and "beat" in vtypes:
            candidates.append("beat")
        if "f_peak" in vtypes:
            candidates.append("f_peak")
        if show_pitch_row:
            candidates.append("pitch")
        if "f_peak" in vtypes or show_pitch_row:
            candidates.append("footer")
        visible = candidates[: max(0, height - 1)]
        show_peaks = "peak" in visible
        show_beats = "beat" in visible
        show_f_peak = "f_peak" in visible
        show_pitch = "pitch" in visible
        show_footer = "footer" in visible
        spectrum_height = max(1, height - len(visible))

        rows: list[Text] = []
        if clock is not None and (show_beats or show_peaks):
            now_us = clock()
            bpm = self._state.beat_state.tempo_bpm()
            beats_label = f"beats ({bpm} BPM):" if bpm is not None else "beats:"
            peaks_label = "peaks:"
            # Drop labels on narrow terminals to keep the strips usable.
            gutter = max(len(beats_label), len(peaks_label)) + 1 if bar_width > 28 else 0
            strip_width = bar_width - gutter
            if show_peaks:
                rows.append(
                    self._gutter_label(
                        peaks_label,
                        gutter,
                        render_peak_strip(
                            width=strip_width,
                            now_us=now_us,
                            recent=self._state.peak_state.recent(),
                            upcoming=self._state.peak_state.upcoming(),
                            loudness=loudness,
                            color=on_color if palette_on else None,
                            playhead_color=text_color if palette_on else None,
                        ),
                        row_bg,
                    )
                )
            if show_beats:
                rows.append(
                    self._gutter_label(
                        beats_label,
                        gutter,
                        render_beat_strip(
                            width=strip_width,
                            now_us=now_us,
                            recent=self._state.beat_state.recent(),
                            upcoming=self._state.beat_state.upcoming(),
                            loudness=loudness,
                            pulse=beat_pulse,
                            marker_color=on_color if palette_on else None,
                            playhead_color=text_color if palette_on else None,
                        ),
                        row_bg,
                    )
                )

        rows.extend(
            render_spectrum(
                magnitudes,
                bar_width,
                spectrum_height,
                loudness,
                peaks,
                beat_pulse=beat_pulse,
                palette_low=palette_low,
                palette_high=palette_high,
                bg_color=bg_color,
                freq_peak_color=text_color,
                freq_peak_column=f_peak_col,
            )
        )

        if show_f_peak:
            markers = [f_peak_marker] if f_peak_marker is not None else []
            rows.append(self._pad_bg(render_freq_cursor_row(bar_width, markers), bar_width, row_bg))
        if show_pitch:
            markers = [pitch_marker] if pitch_marker is not None else []
            rows.append(self._pad_bg(render_freq_cursor_row(bar_width, markers), bar_width, row_bg))
        if show_footer:
            footer = Text()
            for i, (text, color) in enumerate(footer_parts):
                if i:
                    footer.append("   ")
                footer.append(text, style=color)
            rows.append(self._pad_bg(footer, bar_width, row_bg))
        return rows

    @staticmethod
    def _pad_bg(row: Text, width: int, row_bg: str) -> Text:
        """Pad a row to full width and paint the palette background behind it."""
        if row.cell_len < width:
            row.append(" " * (width - row.cell_len))
        if row_bg:
            row.style = row_bg
        return row

    def _build_freq_cursors(
        self,
        width: int,
        pitch_color: str,
        f_peak_color: str,
    ) -> tuple[
        tuple[int, str, str] | None,
        tuple[int, str, str] | None,
        list[tuple[str, str]],
        int | None,
    ]:
        """Build frequency-cursor markers and footer labels for tonal readouts.

        Returns ``(f_peak_marker, pitch_marker, footer_parts, f_peak_column)``
        where each marker is ``(column, glyph, hex_color)`` or ``None``,
        footer_parts are ``(text, hex_color)``, and f_peak_column is the
        dominant-frequency spectrum column (or None). The footer text leads with
        each cursor's glyph so the readout is self-keying.
        """
        state = self._state.visualizer_state
        types = self._state.visualizer_types
        footer: list[tuple[str, str]] = []
        pitch_marker: tuple[int, str, str] | None = None
        f_peak_marker: tuple[int, str, str] | None = None

        f_peak_col: int | None = None
        f_peak_freq = state.f_peak_freq
        if "f_peak" in types and f_peak_freq is not None:
            f_peak_col = freq_to_display_column(f_peak_freq, width)
            if f_peak_col is not None:
                f_peak_marker = (f_peak_col, "△", f_peak_color)
            # Pad to 5 digits (max 20000 Hz) so the pitch label after it stays put.
            footer.append((f"△ f_peak: {f_peak_freq:>5} Hz", f_peak_color))

        note = state.pitch_note
        pitch_freq = state.pitch_freq
        if state.has_pitch:
            assert pitch_freq is not None
            col = freq_to_display_column(pitch_freq, width)
            if col is not None:
                pitch_marker = (col, "▲", pitch_color)
            footer.append((f"▲ pitch: {note}", pitch_color))

        return f_peak_marker, pitch_marker, footer, f_peak_col

    @staticmethod
    def _gutter_label(label: str, gutter: int, strip: Text, row_bg: str) -> Text:
        """Prefix a timeline strip with a left-gutter label and paint the row bg.

        The label is a dim span so the background (set as the row's base style)
        shows behind both the label and the strip without dimming the strip.
        """
        row = Text(style=row_bg)
        if gutter > 0:
            row.append(label.ljust(gutter), style=f"dim {row_bg}".strip())
        row.append_text(strip)
        return row

    def _measure_layout_height(self, layout: Table) -> int:
        """Measure the rendered height of a layout table."""
        lines = 0
        for segment in self._console.render(layout):
            lines += str(segment.text).count("\n")
        return lines

    def _build_layout(self) -> Table:
        """Build the complete UI layout."""
        # Get terminal width and leave 1 char margin to prevent wrapping
        width = self._console.width - 1

        # Paint the grid bg so inter-panel gaps inherit the album background.
        layout_style = self._palette_panel_style()
        layout = Table.grid(expand=False)
        if layout_style is not None:
            layout.style = layout_style
        layout.add_column(width=width)

        # Show server selector if active
        if self._state.show_server_selector:
            selector = self._cached_panel(
                "server_selector",
                (
                    tuple(s.url for s in self._state.available_servers),
                    self._state.selected_server_index,
                    self._state.server_url,
                    self._is_highlighted("selector-up"),
                    self._is_highlighted("selector-down"),
                    self._is_highlighted("selector-enter"),
                ),
                self._build_server_selector_panel,
            )
            layout.add_row(selector)
            return layout

        # Pairing takes priority over the normal layout: it blocks playback
        # until the operator approves this client on the server.
        if self._state.pairing_code:
            pairing = self._cached_panel(
                "pairing",
                (self._state.pairing_code,),
                self._build_pairing_panel,
            )
            layout.add_row(pairing)
            return layout

        narrow = width < 80

        # Now Playing panel
        now_playing = self._cached_panel(
            "now_playing",
            (
                self._state.playback_state,
                self._state.title,
                self._state.artist,
                self._state.album,
                self._state.artwork_generation,
                narrow,
                self._is_highlighted("prev"),
                self._is_highlighted("space"),
                self._is_highlighted("next"),
            ),
            lambda: self._build_now_playing_panel(expand=True),
        )

        # Volume panel
        volume = self._cached_panel(
            "volume",
            (
                self._state.volume,
                self._state.muted,
                self._state.player_volume,
                self._state.player_muted,
                self._state.use_external_volume,
                self._is_highlighted("up"),
                self._is_highlighted("down"),
                self._is_highlighted("mute"),
                self._is_highlighted("group-down"),
                self._is_highlighted("group-up"),
                self._is_highlighted("group-mute"),
            ),
            lambda: self._build_volume_panel(expand=True),
        )

        # Progress bar — only cache when not playing (interpolation needs fresh renders)
        if self._state.playback_state == PlaybackStateType.PLAYING:
            progress = self._build_progress_bar(expand=True)
        else:
            progress = self._cached_panel(
                "progress",
                (self._state.track_progress_ms, self._state.track_duration_ms, width),
                lambda: self._build_progress_bar(expand=True),
            )

        # Bottom panels
        min_rows = 0 if narrow else 5

        playback = self._cached_panel(
            "playback",
            (
                narrow,
                self._state.repeat_mode,
                self._state.shuffle,
                self._is_highlighted("repeat"),
                self._is_highlighted("shuffle"),
            ),
            lambda: self._build_playback_panel(expand=True, min_info_rows=min_rows),
        )

        stream = self._cached_panel(
            "stream",
            (
                narrow,
                self._state.audio_codec,
                self._state.audio_sample_rate,
                self._state.audio_bit_depth,
                self._state.audio_channels,
                self._state.delay_ms,
                self._is_highlighted("delay-"),
                self._is_highlighted("delay+"),
            ),
            lambda: self._build_stream_quality_panel(expand=True, min_info_rows=min_rows),
        )

        server = self._cached_panel(
            "server",
            (
                narrow,
                self._state.connected,
                self._state.server_url,
                self._state.status_message,
                self._state.group_name,
                self._is_highlighted("switch"),
                self._is_highlighted("server"),
            ),
            lambda: self._build_server_panel(expand=True, min_info_rows=min_rows),
        )

        if narrow:
            layout.add_row(now_playing)
            layout.add_row(volume)
        else:
            top_row = Table.grid(expand=True)
            top_row.add_column(ratio=2)
            top_row.add_column(ratio=1)
            top_row.add_row(now_playing, volume)
            layout.add_row(top_row)

        layout.add_row(progress)

        if narrow:
            layout.add_row(playback)
            layout.add_row(stream)
            layout.add_row(server)
        else:
            bottom_row = Table.grid(expand=True)
            bottom_row.add_column(ratio=1)
            bottom_row.add_column(ratio=1)
            bottom_row.add_column(ratio=1)
            bottom_row.add_row(playback, stream, server)
            layout.add_row(bottom_row)

        # Apply panel bg so this row keeps text contrast with the visualizer off.
        quit_line = Text(justify="right", style=layout_style or "")
        if self._palette_active():
            target = "light" if self._state.color_mode == ColorMode.DARK else "dark"
            quit_line.append("t", style=self._shortcut_style("theme"))
            quit_line.append(f" {target} theme  ", style=self._themed("dim"))
        quit_line.append("v", style=self._shortcut_style("visualizer"))
        quit_line.append(" visualizer  ", style=self._themed("dim"))
        quit_line.append("q", style=self._shortcut_style("quit"))
        quit_line.append(" quit  ", style=self._themed("dim"))
        layout.add_row(quit_line)

        # Visualizer: fill remaining terminal space
        if self._state.visualizer_enabled:
            panel_height = self._measure_layout_height(layout)
            remaining = self._console.height - panel_height
            if remaining >= 3:
                for row in self._build_visualizer_rows(remaining):
                    layout.add_row(row)

        return layout

    def add_event(self, _message: str) -> None:
        """Add an event (no-op, events panel removed)."""

    def refresh(self) -> None:
        """Request a coalesced UI refresh."""
        self._dirty = True
        if self._live is None or self._batch_depth > 0:
            return

        self._refresh_event.set()

    @contextmanager
    def batch_update(self) -> Iterator[None]:
        """Delay rendering until a related group of state updates completes."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._dirty:
                self.refresh()

    def set_connected(self, url: str) -> None:
        """Update connection status to connected."""
        self._state.connected = True
        self._state.server_url = url
        self._state.status_message = f"Connected to {url}"
        self.refresh()

    def set_group_name(self, name: str | None) -> None:
        """Update the group name."""
        self._state.group_name = name
        self.refresh()

    def set_disconnected(self, message: str = "Disconnected") -> None:
        """Update connection status to disconnected."""
        self._state.connected = False
        self._state.status_message = message
        self.refresh()

    def set_playback_state(self, state: PlaybackStateType) -> None:
        """Update playback state."""
        # When leaving PLAYING, capture interpolated progress so display doesn't jump
        if (
            self._state.playback_state == PlaybackStateType.PLAYING
            and state != PlaybackStateType.PLAYING
            and self._state.progress_updated_at > 0
            and self._state.track_duration_ms
        ):
            elapsed_ms = (time.monotonic() - self._state.progress_updated_at) * 1000
            interpolated = (self._state.track_progress_ms or 0) + int(elapsed_ms)
            self._state.track_progress_ms = min(self._state.track_duration_ms, interpolated)
            # Reset timestamp so resume starts fresh from captured position
            self._state.progress_updated_at = time.monotonic()

        self._state.playback_state = state
        self.refresh()

    def set_metadata(
        self,
        title: str | None = None,
        artist: str | None = None,
        album: str | None = None,
    ) -> None:
        """Update track metadata."""
        self._state.title = title
        self._state.artist = artist
        self._state.album = album
        self.refresh()

    def set_progress(self, progress_ms: int | None, duration_ms: int | None) -> None:
        """Update track progress."""
        self._state.track_progress_ms = progress_ms
        self._state.track_duration_ms = duration_ms
        self._state.progress_updated_at = time.monotonic()
        self.refresh()

    def clear_progress(self) -> None:
        """Clear track progress completely, preventing any interpolation."""
        self._state.track_progress_ms = None
        self._state.track_duration_ms = None
        self._state.progress_updated_at = 0.0
        self.refresh()

    def set_volume(self, volume: int | None, *, muted: bool | None = None) -> None:
        """Update group volume."""
        if volume is not None:
            self._state.volume = volume
        if muted is not None:
            self._state.muted = muted
        self.refresh()

    def set_player_volume(self, volume: int, *, muted: bool) -> None:
        """Update player volume."""
        self._state.player_volume = volume
        self._state.player_muted = muted
        self.refresh()

    def set_audio_format(
        self, codec: str | None, sample_rate: int, bit_depth: int, channels: int
    ) -> None:
        """Update audio format display."""
        self._state.audio_codec = codec
        self._state.audio_sample_rate = sample_rate
        self._state.audio_bit_depth = bit_depth
        self._state.audio_channels = channels
        self.refresh()

    def set_delay(self, delay_ms: float) -> None:
        """Update the delay display."""
        self._state.delay_ms = delay_ms
        self.refresh()

    def set_repeat_shuffle(
        self,
        repeat_mode: RepeatMode | None,
        shuffle: bool | None,
    ) -> None:
        """Update repeat mode and shuffle state."""
        self._state.repeat_mode = repeat_mode
        self._state.shuffle = shuffle
        self.refresh()

    def set_visualizer_frame(
        self,
        spectrum: list[int] | None,
        loudness: int | None,
        pitch_midi_q88: int | None = None,
        f_peak_freq: int | None = None,
    ) -> None:
        """Update visualizer state with new frame data."""
        if self._state.visualizer_enabled:
            self._state.visualizer_state.update(spectrum, loudness, pitch_midi_q88, f_peak_freq)
            self.refresh()

    def set_visualizer_types(self, types: frozenset[str]) -> None:
        """Record the visualizer types the server negotiated for this stream."""
        self._state.visualizer_types = types
        self.refresh()

    def set_visualizer_enabled(self, enabled: bool) -> None:
        """Update whether the visualizer is enabled."""
        self._state.visualizer_enabled = enabled
        if not enabled:
            self._state.visualizer_state.clear()
            self._state.beat_state.clear()
            self._state.peak_state.clear()
            self._state.visualizer_types = frozenset()
        self.refresh()

    def set_server_clock(self, now_us: Callable[[], int] | None) -> None:
        """Inject the synced-clock callable used by the beat and peak strips."""
        self._state.server_now_us = now_us
        # Rebuild event state with the new clock for proper recent-event windowing.
        self._state.beat_state = BeatState(now_us=now_us)
        self._state.peak_state = PeakState(now_us=now_us)
        self.refresh()

    def record_beat(self, beat: BeatTiming) -> None:
        """Record a beat that has just landed on the client."""
        if not self._state.visualizer_enabled:
            return
        self._state.beat_state.record_beat(beat)
        self.refresh()

    def set_beat_schedule(self, scheduled: list[BeatTiming]) -> None:
        """Update the upcoming beats used by the timeline strip."""
        if not self._state.visualizer_enabled:
            return
        self._state.beat_state.set_schedule(scheduled)
        self.refresh()

    def record_peak(self, peak: PeakEvent) -> None:
        """Record an energy-onset peak that has just landed on the client."""
        if not self._state.visualizer_enabled:
            return
        self._state.peak_state.record_peak(peak)
        self.refresh()

    def set_peak_schedule(self, scheduled: list[PeakEvent]) -> None:
        """Update the upcoming peaks used by the peak strip."""
        if not self._state.visualizer_enabled:
            return
        self._state.peak_state.set_schedule(scheduled)
        self.refresh()

    def clear_beats(self) -> None:
        """Clear all beat state immediately."""
        self._state.beat_state.clear()
        self.refresh()

    def clear_peaks(self) -> None:
        """Clear all peak state immediately."""
        self._state.peak_state.clear()
        self.refresh()

    def show_server_selector(self, servers: list[DiscoveredServer]) -> None:
        """Show the server selector with available servers."""
        self._state.available_servers = servers
        self._state.selected_server_index = 0
        self._state.show_server_selector = True
        self.refresh()

    def hide_server_selector(self) -> None:
        """Hide the server selector."""
        self._state.show_server_selector = False
        self.refresh()

    def is_server_selector_visible(self) -> bool:
        """Check if the server selector is currently visible."""
        return self._state.show_server_selector

    def show_pairing_code(self, code: str | None) -> None:
        """Show (or clear, on ``None``) the derived dynamic pairing PIN."""
        self._state.pairing_code = code
        self.refresh()

    def move_server_selection(self, delta: int) -> None:
        """Move the server selection by delta (-1 for up, +1 for down)."""
        if not self._state.available_servers:
            return
        new_index = self._state.selected_server_index + delta
        self._state.selected_server_index = max(
            0, min(len(self._state.available_servers) - 1, new_index)
        )
        self.refresh()

    def get_selected_server(self) -> DiscoveredServer | None:
        """Get the currently selected server."""
        if not self._state.available_servers:
            return None
        if 0 <= self._state.selected_server_index < len(self._state.available_servers):
            return self._state.available_servers[self._state.selected_server_index]
        return None

    def start(self) -> None:
        """Start the live display."""
        self._console.clear()
        self._update_console_size()
        self._live = Live(
            _RefreshableLayout(self),
            console=self._console,
            auto_refresh=False,
            screen=True,
        )
        self._live.start()
        self._running = True
        self._refresh_task = create_task(self._refresh_loop(), name="sendspin-ui-refresh")
        self.refresh()

    def stop(self) -> None:
        """Stop the live display."""
        self._running = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit."""
        self.stop()
