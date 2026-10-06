"""Visualizer rendering for the Sendspin TUI."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from aiosendspin.models.visualizer import BeatTiming, SpectrumScale
from rich.text import Text

# Spectrum negotiation geometry. Single source of truth: the client/hello
# support payload is built from these (see app._build_visualizer_support), and
# the freq->column cursor math below inverts the same mapping, so the two cannot
# drift apart.
SPECTRUM_N_BINS = 48
SPECTRUM_SCALE: SpectrumScale = "mel"
SPECTRUM_F_MIN = 20
SPECTRUM_F_MAX = 20000

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True, slots=True)
class PeakEvent:
    """A single energy-onset (peak) event with its detector strength."""

    timestamp_us: int
    strength: int  # 0-255


# Unicode block characters for bar rendering (9 levels including space)
_BLOCKS = " ▁▂▃▄▅▆▇█"
_BLOCK_LEVELS = len(_BLOCKS) - 1  # 8

# Interpolation response speed in units per second.
_SMOOTH_RATE_PER_SECOND = 14.0

# Values closer than this are treated as settled, letting the periodic refresh
# loop idle once a frame's interpolation and the peak decay have finished.
_SETTLE_EPSILON = 1e-3

# Peak hold configuration
_PEAK_HOLD_SECONDS = 0.5
_PEAK_FALL_RATE = 0.375  # normalized units per second (≈6 rows/sec at 16 rows)

# Beat flash decay (seconds for pulse to fully fade)
_BEAT_PULSE_DECAY_SECONDS = 0.18
# Beat strip half-window (seconds shown on either side of the playhead)
_BEAT_STRIP_HALF_WINDOW_S = 4.0

# Loudness-to-color tier stops: (loudness_threshold, (R, G, B))
_COLOR_TIERS: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (0x33, 0x55, 0x88)),  # steel blue
    (0.05, (0x33, 0x66, 0x88)),  # blue-teal
    (0.10, (0x33, 0x77, 0x77)),  # teal
    (0.20, (0x44, 0x88, 0x66)),  # sea green
    (0.35, (0x66, 0x88, 0x44)),  # olive
    (0.55, (0x88, 0x77, 0x33)),  # amber
    (0.75, (0x99, 0x55, 0x33)),  # warm brown
]


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to a hex color string for Rich."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linearly interpolate between two RGB colors."""
    return (
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t),
    )


def loudness_to_colors(
    loudness: float,
    palette_low: tuple[int, int, int] | None = None,
    palette_high: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Map a 0.0-1.0 loudness value to (tip_rgb, base_rgb).

    When palette anchors are provided, the tip is lerped between them by
    loudness. Otherwise it follows the static `_COLOR_TIERS`. Base is always
    the tip at 25% brightness.
    """
    loudness = max(0.0, min(1.0, loudness))

    if palette_low is not None and palette_high is not None:
        tip = _lerp_rgb(palette_low, palette_high, loudness)
        base = (tip[0] // 4, tip[1] // 4, tip[2] // 4)
        return tip, base

    for i in range(len(_COLOR_TIERS) - 1):
        t0, c0 = _COLOR_TIERS[i]
        t1, c1 = _COLOR_TIERS[i + 1]
        if loudness <= t1:
            t = (loudness - t0) / (t1 - t0) if t1 > t0 else 0.0
            tip = _lerp_rgb(c0, c1, t)
            base = (tip[0] // 4, tip[1] // 4, tip[2] // 4)
            return tip, base

    tip = _COLOR_TIERS[-1][1]
    base = (tip[0] // 4, tip[1] // 4, tip[2] // 4)
    return tip, base


def midi_to_note_name(midi_q88: int) -> str:
    """Convert an 8.8 fixed-point MIDI value to a note name (e.g. 69.0 -> 'A4')."""
    midi = int(round(midi_q88 / 256.0))
    octave = midi // 12 - 1
    return f"{_NOTE_NAMES[midi % 12]}{octave}"


def _mel(freq_hz: float) -> float:
    """HTK mel scale, matching the server's spectrum binning."""
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def freq_to_display_column(freq_hz: float, width: int) -> int | None:
    """Map a frequency to a spectrum column, inverting the negotiated mel binning.

    Returns None when the frequency is non-positive or the width is empty.
    Frequencies outside [f_min, f_max] clamp to the edge columns.
    """
    if width <= 0 or freq_hz <= 0:
        return None
    f = min(max(freq_hz, SPECTRUM_F_MIN), SPECTRUM_F_MAX)
    m_lo, m_hi = _mel(SPECTRUM_F_MIN), _mel(SPECTRUM_F_MAX)
    frac = (_mel(f) - m_lo) / (m_hi - m_lo) if m_hi > m_lo else 0.0
    bin_index = frac * (SPECTRUM_N_BINS - 1)
    col = int(round(bin_index * width / SPECTRUM_N_BINS))
    return min(width - 1, max(0, col))


class VisualizerState:
    """Stores and smooths visualizer frame data for rendering."""

    def __init__(self) -> None:
        self._spectrum: list[float] = []
        self._spectrum_target: list[float] = []
        self._loudness: float = 0.0
        self._loudness_target: float = 0.0
        self._peaks: list[float] = []
        self._peak_hold_timers: list[float] = []
        self._last_step_monotonic = time.monotonic()
        # Latest tonal readouts (held, not smoothed): pitch and dominant freq.
        self._pitch_midi_q88: int | None = None
        self._f_peak_freq: int | None = None

    def update(
        self,
        spectrum: list[int] | None,
        loudness: int | None,
        pitch_midi_q88: int | None = None,
        f_peak_freq: int | None = None,
    ) -> None:
        """Update with new frame data. Periodic values are uint16 (0-65535).

        Tonal readouts (pitch, f_peak) ride along on the spectrum frame as the
        latest received values; they are held as-is for cursor placement.
        """
        if spectrum is None and loudness is None:
            self.clear()
            return

        if spectrum is not None:
            self._spectrum_target = [v / 65535.0 for v in spectrum]
        if loudness is not None:
            self._loudness_target = loudness / 65535.0
        self._pitch_midi_q88 = pitch_midi_q88
        self._f_peak_freq = f_peak_freq

    def clear(self) -> None:
        """Clear all state immediately."""
        self._spectrum = []
        self._spectrum_target = []
        self._loudness = 0.0
        self._loudness_target = 0.0
        self._peaks = []
        self._peak_hold_timers = []
        self._last_step_monotonic = time.monotonic()
        self._pitch_midi_q88 = None
        self._f_peak_freq = None

    def _step(self) -> None:
        """Advance displayed values toward targets."""
        now = time.monotonic()
        dt = max(0.0, now - self._last_step_monotonic)
        self._last_step_monotonic = now
        if dt <= 0.0:
            return

        alpha = min(1.0, dt * _SMOOTH_RATE_PER_SECOND)

        if len(self._spectrum) != len(self._spectrum_target):
            self._spectrum = list(self._spectrum_target)
        else:
            self._spectrum = [
                current + (target - current) * alpha
                for current, target in zip(self._spectrum, self._spectrum_target, strict=True)
            ]

        self._loudness = self._loudness + (self._loudness_target - self._loudness) * alpha

        # Update peaks
        if len(self._peaks) != len(self._spectrum):
            self._peaks = list(self._spectrum)
            self._peak_hold_timers = [_PEAK_HOLD_SECONDS] * len(self._spectrum)
        else:
            for i, value in enumerate(self._spectrum):
                if value >= self._peaks[i]:
                    self._peaks[i] = value
                    self._peak_hold_timers[i] = _PEAK_HOLD_SECONDS
                elif self._peak_hold_timers[i] > 0:
                    remaining = self._peak_hold_timers[i]
                    self._peak_hold_timers[i] -= dt
                    if self._peak_hold_timers[i] <= 0:
                        decay_dt = dt - remaining
                        self._peaks[i] = max(0.0, self._peaks[i] - _PEAK_FALL_RATE * decay_dt)
                else:
                    self._peaks[i] = max(0.0, self._peaks[i] - _PEAK_FALL_RATE * dt)

    @property
    def is_active(self) -> bool:
        """Whether the visualizer is still animating toward its target.

        True only while the smoothed values have not settled, so the refresh
        loop idles once interpolation and the peak decay finish -- including
        when playback pauses and no further frames arrive.
        """
        if len(self._spectrum) != len(self._spectrum_target):
            return True
        if len(self._peaks) != len(self._spectrum):
            return True
        if abs(self._loudness - self._loudness_target) > _SETTLE_EPSILON:
            return True
        if any(
            abs(current - target) > _SETTLE_EPSILON
            for current, target in zip(self._spectrum, self._spectrum_target, strict=True)
        ):
            return True
        return any(
            abs(peak - bar) > _SETTLE_EPSILON
            for peak, bar in zip(self._peaks, self._spectrum, strict=True)
        )

    def step(self) -> None:
        """Advance displayed values toward targets.

        Call once per render frame before reading spectrum/loudness/peaks.
        """
        self._step()

    def get_spectrum(self) -> list[float]:
        """Return the most recent normalized 0.0-1.0 spectrum values."""
        return list(self._spectrum)

    @property
    def loudness(self) -> float:
        """Return current loudness without per-frame decay."""
        return self._loudness

    def get_peaks(self) -> list[float]:
        """Return the current peak hold heights (0.0-1.0 per bin)."""
        return list(self._peaks)

    @property
    def pitch_note(self) -> str | None:
        """Latest perceived pitch as a note name, or None when not detected."""
        if self._pitch_midi_q88 is None or self._pitch_midi_q88 <= 0:
            return None
        return midi_to_note_name(self._pitch_midi_q88)

    @property
    def pitch_freq(self) -> float | None:
        """Latest perceived pitch as a frequency in Hz, or None when not detected."""
        if self._pitch_midi_q88 is None or self._pitch_midi_q88 <= 0:
            return None
        midi = self._pitch_midi_q88 / 256.0
        return float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))

    @property
    def f_peak_freq(self) -> int | None:
        """Latest dominant frequency in Hz, or None when not detected."""
        if not self._f_peak_freq:
            return None
        return self._f_peak_freq

    @property
    def has_pitch(self) -> bool:
        """Whether a pitch readout is available to display."""
        return self.pitch_note is not None and self.pitch_freq is not None


class BeatState:
    """Stores beat events and produces a decaying pulse intensity.

    The scheduled BeatHandler calls record_beat() exactly when a beat is due
    on the client; the render loop reads pulse_intensity() each frame to drive
    the tip flash. set_schedule() feeds upcoming beats for the timeline strip.
    Each beat carries its downbeat flag so the strip can paint bar starts
    differently.
    """

    def __init__(self, now_us: Callable[[], int] | None = None) -> None:
        """Initialize beat state.

        :param now_us: Function returning the synced server clock in microseconds.
            Optional; falls back to local monotonic time scaled to microseconds.
        """
        self._now_us = now_us
        self._last_beat_monotonic: float | None = None
        self._scheduled: list[BeatTiming] = []
        self._recent: list[BeatTiming] = []

    def record_beat(self, beat: BeatTiming) -> None:
        """Record that a beat has just landed (call from BeatHandler.on_beat)."""
        self._last_beat_monotonic = time.monotonic()
        # Record the beat's actual server timestamp so it lands on the playhead
        # cell instead of drifting one cell right.
        self._recent.append(beat)
        if self._now_us is not None:
            cutoff = self._now_us() - int(_BEAT_STRIP_HALF_WINDOW_S * 1_000_000)
            self._recent = [b for b in self._recent if b.timestamp_us >= cutoff]

    def set_schedule(self, scheduled: list[BeatTiming]) -> None:
        """Set the upcoming beat list (BeatTiming with server-clock timestamps)."""
        self._scheduled = list(scheduled)

    def clear(self) -> None:
        """Clear all beat state immediately."""
        self._last_beat_monotonic = None
        self._scheduled = []
        self._recent = []

    def pulse_intensity(self) -> float:
        """Return current beat pulse intensity (0.0 idle, 1.0 just hit)."""
        if self._last_beat_monotonic is None:
            return 0.0
        elapsed = time.monotonic() - self._last_beat_monotonic
        if elapsed >= _BEAT_PULSE_DECAY_SECONDS:
            return 0.0
        return max(0.0, 1.0 - elapsed / _BEAT_PULSE_DECAY_SECONDS)

    def upcoming(self) -> list[BeatTiming]:
        """Return upcoming beats."""
        return list(self._scheduled)

    def recent(self) -> list[BeatTiming]:
        """Return recent past beats inside the strip window."""
        return list(self._recent)

    @property
    def is_active(self) -> bool:
        """Whether there is current or recent beat activity."""
        return bool(self._scheduled) or self.pulse_intensity() > 0.0

    def tempo_bpm(self) -> int | None:
        """Estimate tempo from the median interval between known beats.

        Uses recent and upcoming beats together. Returns None when fewer than
        two beats are known or the spacing is degenerate.
        """
        times = sorted({b.timestamp_us for b in (*self._recent, *self._scheduled)})
        if len(times) < 2:
            return None
        intervals = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
        if not intervals:
            return None
        intervals.sort()
        median = intervals[len(intervals) // 2]
        if median <= 0:
            return None
        return int(round(60_000_000 / median))


class PeakState:
    """Stores energy-onset (peak) events for the peak timeline strip.

    Mirrors BeatState: the PeakHandler calls record_peak() when a peak is due,
    and set_schedule() feeds upcoming peaks. Each event carries a 0-255 strength
    so the strip can scale glyph height.
    """

    def __init__(self, now_us: Callable[[], int] | None = None) -> None:
        """Initialize peak state.

        :param now_us: Function returning the synced server clock in microseconds,
            used to window recent peaks to the visible strip span.
        """
        self._now_us = now_us
        self._scheduled: list[PeakEvent] = []
        self._recent: list[PeakEvent] = []

    def record_peak(self, peak: PeakEvent) -> None:
        """Record that a peak has just landed (call from PeakHandler.on_peak)."""
        self._recent.append(peak)
        if self._now_us is not None:
            cutoff = self._now_us() - int(_BEAT_STRIP_HALF_WINDOW_S * 1_000_000)
            self._recent = [p for p in self._recent if p.timestamp_us >= cutoff]

    def set_schedule(self, scheduled: list[PeakEvent]) -> None:
        """Set the upcoming peak list (server-clock timestamps)."""
        self._scheduled = list(scheduled)

    def clear(self) -> None:
        """Clear all peak state immediately."""
        self._scheduled = []
        self._recent = []

    def upcoming(self) -> list[PeakEvent]:
        """Return upcoming peaks."""
        return list(self._scheduled)

    def recent(self) -> list[PeakEvent]:
        """Return recent past peaks inside the strip window."""
        return list(self._recent)

    @property
    def is_active(self) -> bool:
        """Whether there are scheduled or recent peaks."""
        return bool(self._scheduled) or bool(self._recent)


def render_beat_strip(
    width: int,
    now_us: int,
    recent: list[BeatTiming],
    upcoming: list[BeatTiming],
    loudness: float,
    pulse: float,
    marker_color: str | None = None,
    playhead_color: str | None = None,
) -> Text:
    """Render a single-row beat timeline strip.

    Past beats appear left of center, upcoming beats appear right of center,
    `│` marks the playhead. Each beat falls onto the closest character cell.
    Downbeats render as ``■``, regular beats as ``●``. If a downbeat and a
    regular beat land on the same cell, the downbeat wins.

    When ``marker_color`` is given (a contrast-guaranteed palette color) it
    paints every beat, and ``playhead_color`` paints the playhead; otherwise the
    colors follow the loudness tiers. Position conveys past vs upcoming.
    """
    if width <= 0:
        return Text("")
    half_us = int(_BEAT_STRIP_HALF_WINDOW_S * 1_000_000)
    if half_us <= 0:
        return Text(" " * width)

    cells = [" "] * width
    styles: list[str | None] = [None] * width
    # Tracks whether a cell already shows a downbeat so a later regular beat
    # doesn't overwrite it.
    is_downbeat_cell = [False] * width
    center = width // 2

    tip, base = loudness_to_colors(loudness)
    if marker_color is not None:
        past_color = upcoming_color = marker_color
        ph_color = playhead_color or marker_color
    else:
        past_color = _rgb_to_hex(*base)
        upcoming_color = _rgb_to_hex(*_lerp_rgb(base, tip, 0.5))
        # Brighten the playhead toward white as the beat pulse peaks.
        ph_color = _rgb_to_hex(
            min(255, int(tip[0] + (255 - tip[0]) * pulse)),
            min(255, int(tip[1] + (255 - tip[1]) * pulse)),
            min(255, int(tip[2] + (255 - tip[2]) * pulse)),
        )

    def place(timestamp_us: int, glyph: str, style: str, *, downbeat: bool) -> None:
        offset_us = timestamp_us - now_us
        if abs(offset_us) > half_us:
            return
        cell = center + int(round((offset_us / half_us) * (width / 2)))
        if not 0 <= cell < width:
            return
        if is_downbeat_cell[cell] and not downbeat:
            return
        cells[cell] = glyph
        styles[cell] = style
        if downbeat:
            is_downbeat_cell[cell] = True

    for beat in recent:
        glyph = "■" if beat.is_downbeat else "●"
        place(beat.timestamp_us, glyph, past_color, downbeat=beat.is_downbeat)
    for beat in upcoming:
        glyph = "■" if beat.is_downbeat else "●"
        place(beat.timestamp_us, glyph, upcoming_color, downbeat=beat.is_downbeat)

    # Playhead overlays whatever was at center. Grows on beat pulse:
    # idle = thin │, mid pulse = heavy ┃, peak pulse = full block █.
    playhead_color = ph_color
    if pulse >= 0.6:
        playhead_glyph = "█"
    elif pulse >= 0.15:
        playhead_glyph = "┃"
    else:
        playhead_glyph = "│"
    cells[center] = playhead_glyph
    styles[center] = playhead_color

    line = Text()
    for ch, style in zip(cells, styles, strict=True):
        if style is None:
            line.append(ch)
        else:
            line.append(ch, style=style)
    return line


def render_peak_strip(
    width: int,
    now_us: int,
    recent: list[PeakEvent],
    upcoming: list[PeakEvent],
    loudness: float,
    color: str | None = None,
    playhead_color: str | None = None,
) -> Text:
    """Render a single-row energy-onset (peak) timeline strip.

    Shares ``render_beat_strip``'s time geometry so it lines up directly beneath
    the beat strip, with the playhead at center so "now" is marked even when no
    beat strip is shown. Each peak's glyph height scales with its 0-255 strength.

    When ``color`` is given (a contrast-guaranteed palette color) it paints every
    onset; otherwise the colors follow the loudness tiers. ``playhead_color``
    colors the center cursor to match the beat strip's playhead.
    """
    if width <= 0:
        return Text("")
    half_us = int(_BEAT_STRIP_HALF_WINDOW_S * 1_000_000)
    if half_us <= 0:
        return Text(" " * width)

    glyphs = _BLOCKS[1:]  # drop the space so even faint onsets show a tick
    cells = [" "] * width
    styles: list[str | None] = [None] * width
    cell_strength = [-1] * width
    center = width // 2

    if color is not None:
        past_color = upcoming_color = color
    else:
        tip, base = loudness_to_colors(loudness)
        past_color = _rgb_to_hex(*base)
        upcoming_color = _rgb_to_hex(*_lerp_rgb(base, tip, 0.5))

    def place(timestamp_us: int, strength: int, color: str) -> None:
        offset_us = timestamp_us - now_us
        if abs(offset_us) > half_us:
            return
        cell = center + int(round((offset_us / half_us) * (width / 2)))
        if not 0 <= cell < width:
            return
        if strength <= cell_strength[cell]:
            return  # a stronger onset already owns this cell
        frac = min(255, max(0, strength)) / 255.0
        cells[cell] = glyphs[int(frac * (len(glyphs) - 1))]
        styles[cell] = color
        cell_strength[cell] = strength

    for peak in recent:
        place(peak.timestamp_us, peak.strength, past_color)
    for peak in upcoming:
        place(peak.timestamp_us, peak.strength, upcoming_color)

    cells[center] = "│"
    styles[center] = playhead_color or color

    line = Text()
    for ch, style in zip(cells, styles, strict=True):
        if style is None:
            line.append(ch)
        else:
            line.append(ch, style=style)
    return line


def render_freq_cursor_row(width: int, markers: list[tuple[int, str, str]]) -> Text:
    """Render arrow cursors pointing at spectrum columns.

    ``markers`` is a list of ``(column, glyph, hex_color)``. Markers are drawn in
    order, so a later one overwrites an earlier one sharing the same cell.
    """
    cells = [" "] * max(0, width)
    styles: list[str | None] = [None] * max(0, width)
    for col, glyph, color in markers:
        if 0 <= col < width:
            cells[col] = glyph
            styles[col] = color
    line = Text()
    for ch, style in zip(cells, styles, strict=True):
        if style is None:
            line.append(ch)
        else:
            line.append(ch, style=style)
    return line


def render_spectrum(
    magnitudes: list[float],
    width: int,
    height: int,
    loudness: float,
    peaks: list[float],
    beat_pulse: float = 0.0,
    palette_low: tuple[int, int, int] | None = None,
    palette_high: tuple[int, int, int] | None = None,
    bg_color: str | None = None,
    freq_peak_color: str = "#ffffff",
    freq_peak_column: int | None = None,
) -> list[Text]:
    """Render spectrum bars as Rich Text lines with loudness-driven color.

    Args:
        magnitudes: Normalized 0.0-1.0 values per frequency bin.
        width: Target width in characters.
        height: Number of text rows (each row = 8 block levels).
        loudness: Normalized 0.0-1.0 loudness for color selection.
        peaks: Normalized 0.0-1.0 peak hold heights per bin.
        beat_pulse: 0.0-1.0 intensity mixing the tip color toward white on beat.
        palette_low: Optional RGB anchor for low-loudness tip color.
        palette_high: Optional RGB anchor for high-loudness tip color.
        bg_color: Optional hex background painted behind every cell.
        freq_peak_color: Hex color for the frequency-peak marker.
        freq_peak_column: Column to highlight as the dominant frequency. When set
            (the server's f_peak), it replaces the local max-bin guess.

    Returns:
        List of Text objects, one per row (top to bottom).
    """
    empty_style = f"on {bg_color}" if bg_color else ""
    if not magnitudes or width <= 0 or height <= 0:
        return [Text(" " * max(0, width), style=empty_style) for _ in range(max(0, height))]

    tip, base = loudness_to_colors(loudness, palette_low, palette_high)
    if beat_pulse > 0.0:
        # Mix tip toward white at peak pulse; keep base untouched so the bottom
        # of the bars doesn't flicker. Capped at 0.5 for a gentle lift instead
        # of a hard strobe.
        flash_t = max(0.0, min(1.0, beat_pulse)) * 0.5
        tip = (
            int(tip[0] + (255 - tip[0]) * flash_t),
            int(tip[1] + (255 - tip[1]) * flash_t),
            int(tip[2] + (255 - tip[2]) * flash_t),
        )

    # Prefer the server's dominant frequency; fall back to the local max bin.
    use_server_peak = freq_peak_column is not None
    freq_peak_bin = (
        -1 if use_server_peak else max(range(len(magnitudes)), key=lambda i: magnitudes[i])
    )

    # Resample magnitudes and peaks to fit width
    n_bins = len(magnitudes)
    bars: list[float] = []
    bar_peaks: list[float] = []
    bar_is_freq_peak: list[bool] = []
    for i in range(width):
        start = i * n_bins / width
        end = (i + 1) * n_bins / width
        start_idx = int(start)
        end_idx = min(int(end), n_bins - 1)

        total = 0.0
        peak_max = 0.0
        is_freq_peak = False
        count = 0
        for j in range(start_idx, end_idx + 1):
            total += magnitudes[j]
            if peaks and j < len(peaks):
                peak_max = max(peak_max, peaks[j])
            if j == freq_peak_bin:
                is_freq_peak = True
            count += 1
        value = total / count if count > 0 else 0.0
        if value > 0.0:
            value = value**0.6
        bars.append(value)
        bar_peaks.append(peak_max**0.6 if peak_max > 0 else 0.0)
        bar_is_freq_peak.append(is_freq_peak)

    if use_server_peak and freq_peak_column is not None and 0 <= freq_peak_column < width:
        bar_is_freq_peak = [i == freq_peak_column for i in range(width)]

    total_levels = height * _BLOCK_LEVELS
    rows: list[Text] = []

    # Pre-compute row colors (vertical gradient from base to tip)
    # Square root curve so bars brighten quickly from the base
    row_colors: list[str] = []
    for row in range(height):
        # Row 0 = top (brightest), row height-1 = bottom (darkest)
        linear_t = 1.0 - row / max(1, height - 1) if height > 1 else 1.0
        t = linear_t**0.5  # sqrt curve: more of the bar sits in brighter range
        rgb = _lerp_rgb(base, tip, t)
        row_colors.append(_rgb_to_hex(*rgb))

    # Peak marker colors
    peak_color = _rgb_to_hex(
        min(255, int(tip[0] * 1.4)),
        min(255, int(tip[1] * 1.4)),
        min(255, int(tip[2] * 1.4)),
    )

    for row_idx in range(height):
        line = Text(style=empty_style)
        row_bottom = (height - 1 - row_idx) * _BLOCK_LEVELS
        color = row_colors[row_idx]
        bar_style = f"{color} on {bg_color}" if bg_color else color

        for bar_idx, value in enumerate(bars):
            level = value * total_levels
            if value > 0.0:
                level = max(level, 1.0)
            fill = level - row_bottom

            # Check for peak marker at this position
            peak_level = bar_peaks[bar_idx] * total_levels
            if peak_level > 0.0:
                peak_level = max(peak_level, 1.0)
            peak_row_pos = peak_level - row_bottom

            if 0 < peak_row_pos <= _BLOCK_LEVELS and fill < peak_row_pos:
                pc = freq_peak_color if bar_is_freq_peak[bar_idx] else peak_color
                line.append("▔", style=pc)
            elif fill >= _BLOCK_LEVELS:
                line.append(_BLOCKS[_BLOCK_LEVELS], style=bar_style)
            elif fill <= 0:
                line.append(" ", style=empty_style)
            else:
                line.append(_BLOCKS[int(fill)], style=bar_style)

        rows.append(line)

    return rows
