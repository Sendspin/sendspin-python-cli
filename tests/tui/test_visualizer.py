"""Tests for the TUI visualizer rendering."""

import time
from unittest.mock import patch

from aiosendspin.models.visualizer import BeatTiming

from sendspin.tui.ui import SendspinUI
from sendspin.tui.visualizer import (
    SPECTRUM_F_MAX,
    SPECTRUM_F_MIN,
    BeatState,
    PeakEvent,
    PeakState,
    VisualizerState,
    freq_to_display_column,
    loudness_to_colors,
    midi_to_note_name,
    render_beat_strip,
    render_peak_strip,
    render_spectrum,
)

# MIDI 69 (A4) in q8.8 fixed point, well above the pitch-detection floor.
_A4_MIDI_Q88 = 69 * 256


def _ui_with_tonal_frame(types: frozenset[str]) -> SendspinUI:
    """A visualizer UI holding both a pitch and an f_peak readout."""
    ui = SendspinUI(0.0, visualizer_enabled=True)
    ui.set_visualizer_frame([30000] * 32, 40000, _A4_MIDI_Q88, 1000)
    ui.set_visualizer_types(types)
    return ui


# --- loudness_to_colors tests ---


def test_loudness_zero_returns_first_tier() -> None:
    tip, base = loudness_to_colors(0.0)
    assert tip == (0x33, 0x55, 0x88)
    assert base == (0x33 // 4, 0x55 // 4, 0x88 // 4)


def test_loudness_full_returns_last_tier() -> None:
    tip, base = loudness_to_colors(1.0)
    assert tip == (0x99, 0x55, 0x33)
    assert base == (0x99 // 4, 0x55 // 4, 0x33 // 4)


def test_loudness_at_tier_boundary() -> None:
    tip, _base = loudness_to_colors(0.20)
    # At 20% we hit sea green exactly
    assert tip == (0x44, 0x88, 0x66)


def test_loudness_between_tiers_interpolates() -> None:
    tip, _base = loudness_to_colors(0.025)
    # Halfway between steel blue and blue-teal
    assert 0x33 <= tip[0] <= 0x33
    assert 0x55 < tip[1] < 0x66
    assert 0x77 < tip[2] <= 0x88


# --- VisualizerState peak hold tests ---


def test_peaks_snap_to_bar_height() -> None:
    state = VisualizerState()
    state.update([32768, 65535, 16384], loudness=32768)
    state.step()
    spectrum = state.get_spectrum()
    peaks = state.get_peaks()
    assert len(peaks) == len(spectrum)
    assert peaks == spectrum


def test_peaks_hold_when_bars_drop() -> None:
    state = VisualizerState()
    state.update([65535, 65535], loudness=32768)
    state.step()
    _ = state.get_spectrum()
    initial_peaks = state.get_peaks()

    state.update([0, 0], loudness=32768)
    state.step()
    _ = state.get_spectrum()
    peaks_after_drop = state.get_peaks()
    assert peaks_after_drop[0] >= initial_peaks[0] * 0.9


def test_peaks_decay_after_hold() -> None:
    state = VisualizerState()
    state.update([65535, 65535], loudness=32768)
    state.step()
    _ = state.get_spectrum()
    _ = state.get_peaks()

    state.update([0, 0], loudness=32768)

    base = time.monotonic()
    call_count = 0

    def advancing_monotonic() -> float:
        nonlocal call_count
        call_count += 1
        return base + 1.0 + call_count * 0.001

    with patch("sendspin.tui.visualizer.time") as mock_time:
        mock_time.monotonic.side_effect = advancing_monotonic
        state.step()
        peaks = state.get_peaks()
    assert peaks[0] < 0.9


def test_peaks_cleared_on_clear() -> None:
    state = VisualizerState()
    state.update([65535], loudness=32768)
    state.step()
    _ = state.get_spectrum()
    _ = state.get_peaks()
    state.clear()
    assert state.get_peaks() == []


# --- render_spectrum tests ---


def test_render_spectrum_returns_correct_row_count() -> None:
    magnitudes = [0.5] * 10
    peaks = [0.8] * 10
    rows = render_spectrum(magnitudes, width=20, height=8, loudness=0.5, peaks=peaks)
    assert len(rows) == 8


def test_render_spectrum_empty_magnitudes() -> None:
    rows = render_spectrum([], width=20, height=4, loudness=0.5, peaks=[])
    assert len(rows) == 4
    for row in rows:
        assert row.plain.strip() == ""


def test_render_spectrum_peak_marker_character() -> None:
    magnitudes = [0.5]
    peaks = [0.9]
    rows = render_spectrum(magnitudes, width=1, height=8, loudness=0.5, peaks=peaks)
    all_chars = "".join(row.plain for row in rows)
    assert "▔" in all_chars


def test_palette_anchors_override_color_tiers() -> None:
    low = (10, 20, 30)
    high = (200, 100, 50)

    tip_low, base_low = loudness_to_colors(0.0, palette_low=low, palette_high=high)
    assert tip_low == low
    assert base_low == (low[0] // 4, low[1] // 4, low[2] // 4)

    tip_high, _ = loudness_to_colors(1.0, palette_low=low, palette_high=high)
    assert tip_high == high

    tip_mid, _ = loudness_to_colors(0.5, palette_low=low, palette_high=high)
    assert tip_mid == (105, 60, 40)


def test_render_spectrum_bg_color_paints_empty_cells() -> None:
    rows = render_spectrum([0.0], width=2, height=2, loudness=0.0, peaks=[0.0], bg_color="#abcdef")
    for row in rows:
        assert str(row.style) == "on #abcdef"


def test_render_spectrum_freq_peak_color_styles_peak_marker() -> None:
    magnitudes = [0.5]
    peaks = [0.9]
    rows = render_spectrum(
        magnitudes,
        width=1,
        height=8,
        loudness=0.5,
        peaks=peaks,
        freq_peak_color="#ff00ff",
    )
    marker_styles = [
        str(span.style)
        for row in rows
        for span in row.spans
        if row.plain[span.start : span.end] == "▔"
    ]
    assert marker_styles == ["#ff00ff"]


def test_render_spectrum_beat_pulse_brightens_tip() -> None:
    """beat_pulse>0 should brighten the tip color (top row) without going full white."""
    magnitudes = [1.0]
    peaks = [0.0]
    no_pulse = render_spectrum(
        magnitudes, width=1, height=2, loudness=0.5, peaks=peaks, beat_pulse=0.0
    )
    full_pulse = render_spectrum(
        magnitudes, width=1, height=2, loudness=0.5, peaks=peaks, beat_pulse=1.0
    )
    no_top_style = no_pulse[0].spans[0].style if no_pulse[0].spans else ""
    full_top_style = full_pulse[0].spans[0].style if full_pulse[0].spans else ""
    assert no_top_style != full_top_style
    # Cap of 0.5 means peak pulse must not reach pure white (#ffffff).
    assert "#ffffff" not in str(full_top_style)


# --- BeatState tests ---


def test_beat_state_idle_pulse_is_zero() -> None:
    state = BeatState()
    assert state.pulse_intensity() == 0.0
    assert state.is_active is False


def test_beat_state_pulse_decays_to_zero() -> None:
    state = BeatState()
    state.record_beat(BeatTiming(0))
    assert state.pulse_intensity() > 0.5

    with patch("sendspin.tui.visualizer.time") as mock_time:
        # 1 second after — way past decay window
        mock_time.monotonic.return_value = time.monotonic() + 1.0
        assert state.pulse_intensity() == 0.0


def test_beat_state_idle_after_pulse_decays() -> None:
    """is_active returns to rest once the pulse decays, so the refresh loop idles."""
    state = BeatState()
    state.record_beat(BeatTiming(0))
    assert state.is_active is True

    with patch("sendspin.tui.visualizer.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1.0
        assert state.is_active is False


def test_beat_state_set_schedule_marks_active() -> None:
    state = BeatState()
    state.set_schedule([BeatTiming(100), BeatTiming(200), BeatTiming(300)])
    assert state.is_active is True
    assert [b.timestamp_us for b in state.upcoming()] == [100, 200, 300]


def test_beat_state_clear_resets() -> None:
    state = BeatState()
    state.record_beat(BeatTiming(0))
    state.set_schedule([BeatTiming(1), BeatTiming(2)])
    state.clear()
    assert state.is_active is False
    assert state.upcoming() == []
    assert state.recent() == []


def test_beat_state_recent_windowed() -> None:
    """Recent beats outside the visible window are pruned."""
    now = [10_000_000_000]
    state = BeatState(now_us=lambda: now[0])
    state.record_beat(BeatTiming(now[0]))
    # Advance clock past the strip window
    now[0] += 10_000_000  # 10s
    state.record_beat(BeatTiming(now[0]))
    assert len(state.recent()) == 1
    assert state.recent()[0].timestamp_us == now[0]


# --- render_beat_strip tests ---


def test_render_beat_strip_playhead_in_center() -> None:
    line = render_beat_strip(width=21, now_us=0, recent=[], upcoming=[], loudness=0.5, pulse=0.0)
    # Idle pulse uses the thin playhead glyph.
    assert line.plain[10] == "│"


def test_render_beat_strip_playhead_grows_on_pulse() -> None:
    mid = render_beat_strip(width=21, now_us=0, recent=[], upcoming=[], loudness=0.5, pulse=0.3)
    peak = render_beat_strip(width=21, now_us=0, recent=[], upcoming=[], loudness=0.5, pulse=1.0)
    assert mid.plain[10] == "┃"
    assert peak.plain[10] == "█"


def test_render_beat_strip_past_and_future_dots() -> None:
    half_s = 4.0
    line = render_beat_strip(
        width=21,
        now_us=0,
        recent=[BeatTiming(-int(half_s * 0.5 * 1_000_000))],
        upcoming=[BeatTiming(int(half_s * 0.5 * 1_000_000))],
        loudness=0.5,
        pulse=0.0,
    )
    # Past beat lands ~25% to the left of center, future beat ~25% right.
    assert line.plain.count("●") == 2
    assert line.plain[5] == "●"
    assert line.plain[15] == "●"


def test_render_beat_strip_downbeat_renders_square() -> None:
    """Downbeats render as a square block (■) instead of a circle (●)."""
    line = render_beat_strip(
        width=21,
        now_us=0,
        recent=[BeatTiming(-2_000_000, is_downbeat=True)],
        upcoming=[BeatTiming(2_000_000, is_downbeat=False)],
        loudness=0.5,
        pulse=0.0,
    )
    assert line.plain.count("■") == 1
    assert line.plain.count("●") == 1
    assert line.plain[5] == "■"
    assert line.plain[15] == "●"


def test_render_beat_strip_downbeat_wins_overlap() -> None:
    """When a regular and a downbeat fall on the same cell, the downbeat keeps it."""
    line = render_beat_strip(
        width=21,
        now_us=0,
        recent=[BeatTiming(-2_000_000, is_downbeat=True)],
        upcoming=[BeatTiming(-2_000_000, is_downbeat=False)],
        loudness=0.5,
        pulse=0.0,
    )
    assert line.plain[5] == "■"
    assert line.plain.count("●") == 0


def test_render_beat_strip_beats_outside_window_dropped() -> None:
    line = render_beat_strip(
        width=21,
        now_us=0,
        recent=[BeatTiming(-100_000_000)],  # 100s in the past
        upcoming=[BeatTiming(100_000_000)],
        loudness=0.5,
        pulse=0.0,
    )
    assert line.plain.count("●") == 0
    assert line.plain.count("■") == 0


# --- BeatState.tempo_bpm tests ---


def test_tempo_bpm_even_spacing() -> None:
    """Beats 0.5s apart yield 120 BPM."""
    state = BeatState()
    state.set_schedule([BeatTiming(0), BeatTiming(500_000), BeatTiming(1_000_000)])
    assert state.tempo_bpm() == 120


def test_tempo_bpm_needs_two_beats() -> None:
    state = BeatState()
    assert state.tempo_bpm() is None
    state.set_schedule([BeatTiming(0)])
    assert state.tempo_bpm() is None


# --- PeakState tests ---


def test_peak_state_set_schedule_marks_active() -> None:
    state = PeakState()
    assert state.is_active is False
    state.set_schedule([PeakEvent(100, 200), PeakEvent(200, 50)])
    assert state.is_active is True
    assert [p.timestamp_us for p in state.upcoming()] == [100, 200]


def test_peak_state_recent_windowed() -> None:
    """Recent peaks outside the visible window are pruned."""
    now = [10_000_000_000]
    state = PeakState(now_us=lambda: now[0])
    state.record_peak(PeakEvent(now[0], 100))
    now[0] += 10_000_000  # 10s, past the strip window
    state.record_peak(PeakEvent(now[0], 100))
    assert len(state.recent()) == 1
    assert state.recent()[0].timestamp_us == now[0]


# --- render_peak_strip tests ---


def test_render_peak_strip_marker_placement() -> None:
    line = render_peak_strip(
        width=21,
        now_us=0,
        recent=[PeakEvent(-2_000_000, 200)],
        upcoming=[],
        loudness=0.5,
    )
    # A peak 2s in the past lands ~25% left of center (cell 5 of width 21).
    assert line.plain[5] != " "


def test_render_peak_strip_marks_playhead_at_center() -> None:
    """The peak strip draws the now-cursor at center so 'now' is always marked."""
    line = render_peak_strip(
        width=21,
        now_us=0,
        recent=[],
        upcoming=[],
        loudness=0.5,
    )
    assert line.plain[10] == "│"


def test_render_peak_strip_strength_scales_glyph_height() -> None:
    """A stronger onset draws a taller block glyph than a weaker one."""
    ramp = "▁▂▃▄▅▆▇█"
    line = render_peak_strip(
        width=21,
        now_us=0,
        recent=[PeakEvent(-2_000_000, 10)],
        upcoming=[PeakEvent(2_000_000, 250)],
        loudness=0.5,
    )
    weak = line.plain[5]
    strong = line.plain[15]
    assert ramp.index(strong) > ramp.index(weak)


# --- pitch / frequency helper tests ---


def test_midi_to_note_name_a4() -> None:
    assert midi_to_note_name(69 * 256) == "A4"


def test_midi_to_note_name_rounds_to_nearest_semitone() -> None:
    # 69.6 in 8.8 fixed-point rounds up to MIDI 70 -> A#4.
    assert midi_to_note_name(int(69.6 * 256)) == "A#4"


def test_freq_to_display_column_clamps_to_endpoints() -> None:
    assert freq_to_display_column(SPECTRUM_F_MIN, width=48) == 0
    assert freq_to_display_column(SPECTRUM_F_MAX, width=48) == 47


def test_freq_to_display_column_none_for_nonpositive() -> None:
    assert freq_to_display_column(0, width=48) is None


def test_render_spectrum_uses_server_freq_peak_column() -> None:
    """freq_peak_column overrides the local max bin for the highlight marker."""
    # Local max is bin 0, but the server says the dominant column is the last one.
    magnitudes = [1.0, 0.1, 0.1]
    peaks = [1.0, 1.0, 1.0]
    rows = render_spectrum(
        magnitudes,
        width=3,
        height=8,
        loudness=0.5,
        peaks=peaks,
        freq_peak_color="#ff00ff",
        freq_peak_column=2,
    )
    highlighted_cols = {
        span.start
        for row in rows
        for span in row.spans
        if str(span.style) == "#ff00ff" and row.plain[span.start : span.end] == "▔"
    }
    assert highlighted_cols == {2}


def test_render_beat_strip_uses_palette_colors() -> None:
    """Given palette colors, beats use the marker color and the playhead its own."""
    line = render_beat_strip(
        width=21,
        now_us=0,
        recent=[BeatTiming(-2_000_000)],
        upcoming=[],
        loudness=0.5,
        pulse=0.0,
        marker_color="#abcdef",
        playhead_color="#123456",
    )
    beat_styles = {
        str(span.style) for span in line.spans if line.plain[span.start : span.end] == "●"
    }
    playhead_styles = {
        str(span.style) for span in line.spans if line.plain[span.start : span.end] == "│"
    }
    assert beat_styles == {"#abcdef"}
    assert playhead_styles == {"#123456"}


def test_render_peak_strip_uses_palette_color() -> None:
    line = render_peak_strip(
        width=21,
        now_us=0,
        recent=[PeakEvent(-2_000_000, 200)],
        upcoming=[],
        loudness=0.5,
        color="#abcdef",
    )
    styles = {str(span.style) for span in line.spans if line.plain[span.start : span.end] != " "}
    assert styles == {"#abcdef"}


# --- tonal cursor row tests ---


def test_f_peak_arrow_hidden_when_not_negotiated() -> None:
    """f_peak is type-gated: its data without the negotiated type renders nothing."""
    rows = _ui_with_tonal_frame(frozenset({"pitch"}))._build_visualizer_rows(10)
    plain = "".join(row.plain for row in rows)
    assert "△" not in plain


def test_pitch_arrow_shown_when_not_negotiated() -> None:
    """Pitch is data-gated, not type-gated: a confident readout always renders."""
    rows = _ui_with_tonal_frame(frozenset({"f_peak"}))._build_visualizer_rows(10)
    plain = "".join(row.plain for row in rows)
    assert "▲" in plain


def test_negotiated_pitch_row_keeps_layout_stable() -> None:
    """A negotiated pitch reserves its row, so a lost readout doesn't shift rows."""
    types = frozenset({"f_peak", "pitch"})

    with_pitch = SendspinUI(0.0, visualizer_enabled=True)
    with_pitch.set_visualizer_frame([30000] * 32, 40000, _A4_MIDI_Q88, 1000)
    with_pitch.set_visualizer_types(types)
    rows_with_pitch = with_pitch._build_visualizer_rows(10)

    no_pitch = SendspinUI(0.0, visualizer_enabled=True)
    no_pitch.set_visualizer_frame([30000] * 32, 40000, 0, 1000)  # no pitch readout
    no_pitch.set_visualizer_types(types)
    rows_no_pitch = no_pitch._build_visualizer_rows(10)

    f_peak_confident = next(i for i, r in enumerate(rows_with_pitch) if "△" in r.plain)
    f_peak_silent = next(i for i, r in enumerate(rows_no_pitch) if "△" in r.plain)
    assert f_peak_confident == f_peak_silent


def test_pitch_arrow_on_separate_line_below_f_peak() -> None:
    """With both negotiated, the f_peak arrow gets its own line above pitch's."""
    rows = _ui_with_tonal_frame(frozenset({"f_peak", "pitch"}))._build_visualizer_rows(10)
    f_peak_row = next(i for i, row in enumerate(rows) if "△" in row.plain)
    pitch_row = next(i for i, row in enumerate(rows) if "▲" in row.plain)
    assert f_peak_row < pitch_row
    assert "▲" not in rows[f_peak_row].plain


# --- VisualizerState activity tests ---


def test_visualizer_state_idle_without_data() -> None:
    state = VisualizerState()
    assert state.is_active is False


def test_visualizer_state_settles_to_idle_after_frame() -> None:
    """is_active goes False once interpolation settles, so the refresh loop idles.

    Previously this stayed True forever because the target list was non-empty,
    so a paused stream kept the whole layout re-rendering at the frame interval.
    """
    state = VisualizerState()
    state.update([65535, 65535], loudness=65535)
    assert state.is_active is True

    with patch("sendspin.tui.visualizer.time") as mock_time:
        base = time.monotonic()
        for i in range(1, 200):
            mock_time.monotonic.return_value = base + i * 0.05
            state.step()

    assert state.is_active is False
    assert state.get_spectrum() == [1.0, 1.0]
