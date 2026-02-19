"""
MIDI Feature Extraction - LLM-driven development

Each feature is a function: midi -> float or list[float]
Features are registered and composed automatically.
"""
from __future__ import annotations

import mido
from pathlib import Path
from typing import Callable
import numpy as np

# Feature registry
FEATURES: dict[str, Callable] = {}

def feature(fn: Callable) -> Callable:
    """Decorator to register a feature function."""
    FEATURES[fn.__name__] = fn
    return fn

def load_midi(path: str | Path) -> mido.MidiFile:
    """Load a MIDI file."""
    return mido.MidiFile(path)

def get_notes(midi: mido.MidiFile) -> list[tuple[int, int, float, float]]:
    """Extract all notes as (pitch, velocity, start_time_sec, duration_sec)."""
    notes = []
    for track in midi.tracks:
        abs_time = 0  # in ticks
        active = {}   # pitch -> (velocity, start_tick)
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (msg.velocity, abs_time)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    vel, start_tick = active.pop(msg.note)
                    start_sec = mido.tick2second(start_tick, midi.ticks_per_beat, get_tempo(midi))
                    end_sec = mido.tick2second(abs_time, midi.ticks_per_beat, get_tempo(midi))
                    notes.append((msg.note, vel, start_sec, end_sec - start_sec))
    return sorted(notes, key=lambda n: n[2])  # sort by start time

def get_notes_by_track(midi: mido.MidiFile) -> list[list[tuple[int, int, float, float]]]:
    """Extract notes per track as list of [(pitch, velocity, start_sec, duration_sec), ...]."""
    tempo = get_tempo(midi)
    all_tracks = []
    for track in midi.tracks:
        notes = []
        abs_time = 0
        active = {}
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (msg.velocity, abs_time)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    vel, start_tick = active.pop(msg.note)
                    start_sec = mido.tick2second(start_tick, midi.ticks_per_beat, tempo)
                    end_sec = mido.tick2second(abs_time, midi.ticks_per_beat, tempo)
                    notes.append((msg.note, vel, start_sec, end_sec - start_sec))
        all_tracks.append(sorted(notes, key=lambda n: n[2]))
    return all_tracks

def get_tempo(midi: mido.MidiFile) -> int:
    """Get tempo in microseconds per beat (default 500000 = 120 BPM)."""
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000

def get_drum_notes(midi: mido.MidiFile) -> list[tuple[int, int, float, float]]:
    """Extract drum notes (channel 9) as (note, velocity, start_sec, duration_sec)."""
    notes = []
    for track in midi.tracks:
        abs_time = 0
        active = {}
        for msg in track:
            abs_time += msg.time
            if not hasattr(msg, 'channel') or msg.channel != 9:
                continue
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (msg.velocity, abs_time)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    vel, start_tick = active.pop(msg.note)
                    start_sec = mido.tick2second(start_tick, midi.ticks_per_beat, get_tempo(midi))
                    end_sec = mido.tick2second(abs_time, midi.ticks_per_beat, get_tempo(midi))
                    notes.append((msg.note, vel, start_sec, end_sec - start_sec))
    return sorted(notes, key=lambda n: n[2])

def get_bars(midi: mido.MidiFile) -> float:
    """Get total number of bars (assuming 4/4)."""
    bpm = mido.tempo2bpm(get_tempo(midi))
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4
    return midi.length / seconds_per_bar if seconds_per_bar > 0 else 0.0

def shannon_entropy(values: list) -> float:
    """Compute Shannon entropy of a discrete distribution."""
    if not values:
        return 0.0
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    total = len(values)
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

# --- FEATURES (add below, each decorated with @feature) ---

# === Structure & Form ===

@feature
def note_count(midi: mido.MidiFile) -> int:
    """Total number of notes."""
    return len(get_notes(midi))

@feature  
def duration_seconds(midi: mido.MidiFile) -> float:
    """Total duration in seconds."""
    return midi.length

@feature
def track_count(midi: mido.MidiFile) -> int:
    """Number of MIDI tracks."""
    return len(midi.tracks)

@feature
def silence_ratio(midi: mido.MidiFile) -> float:
    """Fraction of time with no notes sounding."""
    notes = get_notes(midi)
    if not notes:
        return 1.0
    total_dur = midi.length
    if total_dur == 0:
        return 0.0
    # Build list of (start, end) and merge overlapping
    intervals = [(n[2], n[2] + n[3]) for n in notes]
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    sounding_time = sum(e - s for s, e in merged)
    return 1.0 - (sounding_time / total_dur)

# === Rhythm & Timing ===

@feature
def tempo_bpm(midi: mido.MidiFile) -> float:
    """Tempo in BPM from first tempo message."""
    return mido.tempo2bpm(get_tempo(midi))

@feature
def note_density(midi: mido.MidiFile) -> float:
    """Notes per second."""
    dur = midi.length
    if dur == 0:
        return 0.0
    return len(get_notes(midi)) / dur

@feature
def avg_note_duration(midi: mido.MidiFile) -> float:
    """Mean duration of all notes in seconds."""
    notes = get_notes(midi)
    if not notes:
        return 0.0
    return np.mean([n[3] for n in notes])

@feature
def note_duration_variance(midi: mido.MidiFile) -> float:
    """Variance in note durations."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    return np.var([n[3] for n in notes])

# === Pitch & Melody ===

@feature
def pitch_range(midi: mido.MidiFile) -> int:
    """Highest - lowest MIDI note."""
    notes = get_notes(midi)
    if not notes:
        return 0
    pitches = [n[0] for n in notes]
    return max(pitches) - min(pitches)

@feature
def avg_pitch(midi: mido.MidiFile) -> float:
    """Mean MIDI note number."""
    notes = get_notes(midi)
    if not notes:
        return 0.0
    return np.mean([n[0] for n in notes])

@feature
def pitch_std(midi: mido.MidiFile) -> float:
    """Standard deviation of pitches."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    return np.std([n[0] for n in notes])

@feature
def unique_pitches(midi: mido.MidiFile) -> int:
    """Number of distinct pitches used."""
    notes = get_notes(midi)
    return len(set(n[0] for n in notes))

@feature
def pitch_class_histogram(midi: mido.MidiFile) -> list[float]:
    """12-element normalized vector of pitch class frequencies."""
    notes = get_notes(midi)
    hist = np.zeros(12)
    for n in notes:
        hist[n[0] % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.tolist()

@feature
def melodic_interval_avg(midi: mido.MidiFile) -> float:
    """Mean absolute interval size between consecutive notes."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    intervals = [abs(notes[i+1][0] - notes[i][0]) for i in range(len(notes)-1)]
    return np.mean(intervals)

@feature
def melodic_interval_std(midi: mido.MidiFile) -> float:
    """Standard deviation of melodic intervals."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    intervals = [abs(notes[i+1][0] - notes[i][0]) for i in range(len(notes)-1)]
    return np.std(intervals)

@feature
def contour_direction(midi: mido.MidiFile) -> float:
    """Ratio of ascending intervals (1.0 = all up, 0.0 = all down, 0.5 = balanced)."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.5
    intervals = [notes[i+1][0] - notes[i][0] for i in range(len(notes)-1)]
    non_zero = [i for i in intervals if i != 0]
    if not non_zero:
        return 0.5
    ascending = sum(1 for i in non_zero if i > 0)
    return ascending / len(non_zero)

@feature
def leap_ratio(midi: mido.MidiFile) -> float:
    """Fraction of intervals > 4 semitones."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    intervals = [abs(notes[i+1][0] - notes[i][0]) for i in range(len(notes)-1)]
    leaps = sum(1 for i in intervals if i > 4)
    return leaps / len(intervals)

@feature
def stepwise_motion(midi: mido.MidiFile) -> float:
    """Fraction of intervals <= 2 semitones."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    intervals = [abs(notes[i+1][0] - notes[i][0]) for i in range(len(notes)-1)]
    steps = sum(1 for i in intervals if i <= 2)
    return steps / len(intervals)

# === Harmony & Chords ===

def get_polyphony_at_times(midi: mido.MidiFile) -> list[int]:
    """Get polyphony count at each note onset."""
    notes = get_notes(midi)
    if not notes:
        return []
    # For each note onset, count how many notes are sounding
    events = []
    for pitch, vel, start, dur in notes:
        events.append((start, 1))  # note on
        events.append((start + dur, -1))  # note off
    events.sort(key=lambda x: (x[0], -x[1]))  # offs before ons at same time
    
    polyphony = []
    current = 0
    onset_times = set(n[2] for n in notes)
    last_time = None
    for time, delta in events:
        if delta == 1 and time != last_time:
            # Record polyphony at this onset (after applying previous offs)
            polyphony.append(current + 1)
            last_time = time
        current += delta
    return polyphony

@feature
def simultaneous_notes_avg(midi: mido.MidiFile) -> float:
    """Average polyphony (notes sounding together at onsets)."""
    poly = get_polyphony_at_times(midi)
    if not poly:
        return 0.0
    return np.mean(poly)

@feature
def simultaneous_notes_max(midi: mido.MidiFile) -> int:
    """Maximum polyphony."""
    poly = get_polyphony_at_times(midi)
    if not poly:
        return 0
    return max(poly)

# === Dynamics & Velocity ===

@feature
def avg_velocity(midi: mido.MidiFile) -> float:
    """Mean note velocity."""
    notes = get_notes(midi)
    if not notes:
        return 0.0
    return np.mean([n[1] for n in notes])

@feature
def velocity_std(midi: mido.MidiFile) -> float:
    """Velocity standard deviation."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    return np.std([n[1] for n in notes])

@feature
def velocity_range(midi: mido.MidiFile) -> int:
    """Max - min velocity."""
    notes = get_notes(midi)
    if not notes:
        return 0
    vels = [n[1] for n in notes]
    return max(vels) - min(vels)

@feature
def dynamic_contour(midi: mido.MidiFile) -> float:
    """Slope of velocity over time (-1 to 1, positive = getting louder)."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    times = np.array([n[2] for n in notes])
    vels = np.array([n[1] for n in notes])
    if times[-1] == times[0]:
        return 0.0
    # Normalize time to [0, 1]
    times_norm = (times - times[0]) / (times[-1] - times[0])
    # Linear regression slope
    slope = np.corrcoef(times_norm, vels)[0, 1]
    return 0.0 if np.isnan(slope) else slope

@feature
def accent_ratio(midi: mido.MidiFile) -> float:
    """Ratio of notes with velocity > 100."""
    notes = get_notes(midi)
    if not notes:
        return 0.0
    accents = sum(1 for n in notes if n[1] > 100)
    return accents / len(notes)

# === Texture & Orchestration ===

@feature
def channel_count(midi: mido.MidiFile) -> int:
    """Number of active MIDI channels."""
    channels = set()
    for track in midi.tracks:
        for msg in track:
            if hasattr(msg, 'channel'):
                channels.add(msg.channel)
    return len(channels)

@feature
def program_changes(midi: mido.MidiFile) -> int:
    """Number of program change messages."""
    return sum(1 for track in midi.tracks for msg in track if msg.type == 'program_change')

@feature
def instruments_used(midi: mido.MidiFile) -> list[int]:
    """List of GM program numbers used."""
    programs = set()
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'program_change':
                programs.add(msg.program)
    return sorted(programs)

@feature
def bass_presence(midi: mido.MidiFile) -> float:
    """Fraction of notes in low register (< MIDI 48 / C3)."""
    notes = get_notes(midi)
    if not notes:
        return 0.0
    bass = sum(1 for n in notes if n[0] < 48)
    return bass / len(notes)

@feature
def treble_presence(midi: mido.MidiFile) -> float:
    """Fraction of notes in high register (> MIDI 72 / C5)."""
    notes = get_notes(midi)
    if not notes:
        return 0.0
    treble = sum(1 for n in notes if n[0] > 72)
    return treble / len(notes)

@feature
def register_spread(midi: mido.MidiFile) -> float:
    """Standard deviation of pitch, normalized by range."""
    notes = get_notes(midi)
    if len(notes) < 2:
        return 0.0
    pitches = [n[0] for n in notes]
    return np.std(pitches)

# === Genre/Style Indicators ===

@feature
def note_density_high(midi: mido.MidiFile) -> bool:
    """Whether note density > 8 notes/sec."""
    return note_density(midi) > 8

@feature
def chord_heavy(midi: mido.MidiFile) -> bool:
    """Whether avg polyphony > 3."""
    return simultaneous_notes_avg(midi) > 3

@feature
def monophonic(midi: mido.MidiFile) -> bool:
    """Whether avg polyphony ~1 (< 1.2)."""
    return simultaneous_notes_avg(midi) < 1.2

# === Per-Track Features ===

@feature
def note_count_per_track(midi: mido.MidiFile) -> list[int]:
    """Note count for each track."""
    return [len(notes) for notes in get_notes_by_track(midi)]

@feature
def note_density_per_track(midi: mido.MidiFile) -> list[float]:
    """Notes per second for each track."""
    dur = midi.length
    if dur == 0:
        return [0.0] * len(midi.tracks)
    return [len(notes) / dur for notes in get_notes_by_track(midi)]

@feature
def avg_pitch_per_track(midi: mido.MidiFile) -> list[float]:
    """Mean pitch for each track."""
    result = []
    for notes in get_notes_by_track(midi):
        if notes:
            result.append(np.mean([n[0] for n in notes]))
        else:
            result.append(0.0)
    return result

@feature
def pitch_range_per_track(midi: mido.MidiFile) -> list[int]:
    """Pitch range (max - min) for each track."""
    result = []
    for notes in get_notes_by_track(midi):
        if notes:
            pitches = [n[0] for n in notes]
            result.append(max(pitches) - min(pitches))
        else:
            result.append(0)
    return result

@feature
def avg_velocity_per_track(midi: mido.MidiFile) -> list[float]:
    """Mean velocity for each track."""
    result = []
    for notes in get_notes_by_track(midi):
        if notes:
            result.append(np.mean([n[1] for n in notes]))
        else:
            result.append(0.0)
    return result

@feature
def avg_duration_per_track(midi: mido.MidiFile) -> list[float]:
    """Mean note duration for each track."""
    result = []
    for notes in get_notes_by_track(midi):
        if notes:
            result.append(np.mean([n[3] for n in notes]))
        else:
            result.append(0.0)
    return result

@feature
def unique_pitches_per_track(midi: mido.MidiFile) -> list[int]:
    """Number of distinct pitches per track."""
    return [len(set(n[0] for n in notes)) for notes in get_notes_by_track(midi)]

@feature
def polyphony_per_track(midi: mido.MidiFile) -> list[float]:
    """Average polyphony per track."""
    result = []
    for notes in get_notes_by_track(midi):
        if not notes:
            result.append(0.0)
            continue
        events = []
        for pitch, vel, start, dur in notes:
            events.append((start, 1))
            events.append((start + dur, -1))
        events.sort(key=lambda x: (x[0], -x[1]))
        poly = []
        current = 0
        last_time = None
        for time, delta in events:
            if delta == 1 and time != last_time:
                poly.append(current + 1)
                last_time = time
            current += delta
        result.append(np.mean(poly) if poly else 0.0)
    return result

# === Drums & Percussion ===

# GM drum note mappings
KICK = {35, 36}
SNARE = {38, 40}
HIHAT_CLOSED = {42}
HIHAT_PEDAL = {44}
HIHAT_OPEN = {46}
HIHAT_ALL = HIHAT_CLOSED | HIHAT_PEDAL | HIHAT_OPEN
RIDE = {51, 59}
CRASH = {49, 57}
TOMS = {41, 43, 45, 47, 48, 50}

@feature
def kick_density(midi: mido.MidiFile) -> float:
    """Kick drum hits per bar."""
    drums = get_drum_notes(midi)
    bars = get_bars(midi)
    if bars == 0:
        return 0.0
    kicks = sum(1 for n in drums if n[0] in KICK)
    return kicks / bars

@feature
def snare_density(midi: mido.MidiFile) -> float:
    """Snare hits per bar."""
    drums = get_drum_notes(midi)
    bars = get_bars(midi)
    if bars == 0:
        return 0.0
    snares = sum(1 for n in drums if n[0] in SNARE)
    return snares / bars

@feature
def hihat_density(midi: mido.MidiFile) -> float:
    """Hi-hat hits per bar."""
    drums = get_drum_notes(midi)
    bars = get_bars(midi)
    if bars == 0:
        return 0.0
    hihats = sum(1 for n in drums if n[0] in HIHAT_ALL)
    return hihats / bars

@feature
def kick_snare_ratio(midi: mido.MidiFile) -> float:
    """Ratio of kicks to snares (>1 = more kick, <1 = more snare)."""
    drums = get_drum_notes(midi)
    kicks = sum(1 for n in drums if n[0] in KICK)
    snares = sum(1 for n in drums if n[0] in SNARE)
    if snares == 0:
        return float(kicks) if kicks > 0 else 0.0
    return kicks / snares

@feature
def drum_elements_used(midi: mido.MidiFile) -> int:
    """Count of distinct percussion note numbers."""
    drums = get_drum_notes(midi)
    return len(set(n[0] for n in drums))

@feature
def hihat_open_closed_ratio(midi: mido.MidiFile) -> float:
    """Ratio of open to closed hi-hats (higher = more open)."""
    drums = get_drum_notes(midi)
    opens = sum(1 for n in drums if n[0] in HIHAT_OPEN)
    closed = sum(1 for n in drums if n[0] in HIHAT_CLOSED)
    if closed == 0:
        return float(opens) if opens > 0 else 0.0
    return opens / closed

@feature
def ride_vs_hihat(midi: mido.MidiFile) -> float:
    """Ratio of ride to hi-hat usage."""
    drums = get_drum_notes(midi)
    rides = sum(1 for n in drums if n[0] in RIDE)
    hihats = sum(1 for n in drums if n[0] in HIHAT_ALL)
    total = rides + hihats
    if total == 0:
        return 0.5
    return rides / total

@feature
def crash_frequency(midi: mido.MidiFile) -> float:
    """Crash cymbal hits per bar."""
    drums = get_drum_notes(midi)
    bars = get_bars(midi)
    if bars == 0:
        return 0.0
    crashes = sum(1 for n in drums if n[0] in CRASH)
    return crashes / bars

@feature
def tom_usage(midi: mido.MidiFile) -> float:
    """Tom hits per bar."""
    drums = get_drum_notes(midi)
    bars = get_bars(midi)
    if bars == 0:
        return 0.0
    toms = sum(1 for n in drums if n[0] in TOMS)
    return toms / bars

@feature
def percussion_layers(midi: mido.MidiFile) -> float:
    """Average simultaneous percussion voices at drum onsets."""
    drums = get_drum_notes(midi)
    if not drums:
        return 0.0
    events = []
    for note, vel, start, dur in drums:
        events.append((start, 1))
        events.append((start + dur, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    
    layers = []
    current = 0
    last_time = None
    for time, delta in events:
        if delta == 1 and time != last_time:
            layers.append(current + 1)
            last_time = time
        current += delta
    return np.mean(layers) if layers else 0.0

# === Hard Features (no ML) ===

# Krumhansl-Schmuckler key profiles
# Major and minor profiles from Krumhansl & Kessler (1982)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def _key_correlations(pitch_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute correlations with all 12 major and 12 minor key profiles."""
    major_corrs = np.zeros(12)
    minor_corrs = np.zeros(12)
    for i in range(12):
        rotated = np.roll(pitch_hist, -i)
        major_corrs[i] = np.corrcoef(rotated, KS_MAJOR)[0, 1]
        minor_corrs[i] = np.corrcoef(rotated, KS_MINOR)[0, 1]
    return major_corrs, minor_corrs

@feature
def detected_key(midi: mido.MidiFile) -> int:
    """Most likely key (0-11 = C-B major, 12-23 = C-B minor)."""
    hist = np.array(pitch_class_histogram(midi))
    if hist.sum() == 0:
        return 0
    major_corrs, minor_corrs = _key_correlations(hist)
    best_major = np.argmax(major_corrs)
    best_minor = np.argmax(minor_corrs)
    if major_corrs[best_major] >= minor_corrs[best_minor]:
        return int(best_major)
    return int(best_minor + 12)

@feature
def key_clarity(midi: mido.MidiFile) -> float:
    """Krumhansl-Schmuckler correlation strength (0-1). Higher = clearer key."""
    hist = np.array(pitch_class_histogram(midi))
    if hist.sum() == 0:
        return 0.0
    major_corrs, minor_corrs = _key_correlations(hist)
    # Handle NaN from zero-variance histograms
    all_corrs = np.concatenate([major_corrs, minor_corrs])
    all_corrs = np.nan_to_num(all_corrs, nan=0.0)
    return float(np.max(all_corrs))

@feature
def voice_independence(midi: mido.MidiFile) -> float:
    """
    Measure of melodic independence between tracks (0-1).
    Low correlation = independent voices (counterpoint).
    Uses pitch contour correlation between track pairs.
    """
    tracks = get_notes_by_track(midi)
    # Filter to tracks with enough notes
    melodic_tracks = []
    for notes in tracks:
        if len(notes) >= 10:
            # Extract pitch sequence
            pitches = [n[0] for n in notes]
            melodic_tracks.append(pitches)
    
    if len(melodic_tracks) < 2:
        return 0.5  # undefined
    
    # Compute pairwise correlations of pitch contours (differences)
    correlations = []
    for i in range(len(melodic_tracks)):
        for j in range(i + 1, len(melodic_tracks)):
            # Resample to same length for comparison
            len_common = min(len(melodic_tracks[i]), len(melodic_tracks[j]))
            p1 = np.array(melodic_tracks[i][:len_common])
            p2 = np.array(melodic_tracks[j][:len_common])
            if len(p1) > 1:
                # Use contour (intervals) rather than raw pitch
                c1 = np.diff(p1)
                c2 = np.diff(p2)
                if np.std(c1) > 0 and np.std(c2) > 0:
                    corr = abs(np.corrcoef(c1, c2)[0, 1])
                    correlations.append(corr)
    
    if not correlations:
        return 0.5
    # Low mean correlation = high independence, so invert
    return 1.0 - float(np.mean(correlations))

def _get_interval_ngrams(pitches: list[int], n: int = 3) -> list[tuple]:
    """Extract n-grams of melodic intervals (transposition invariant)."""
    if len(pitches) < n + 1:
        return []
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
    return [tuple(intervals[i:i+n]) for i in range(len(intervals) - n + 1)]

@feature
def motif_recurrence(midi: mido.MidiFile) -> float:
    """
    Frequency of repeated short melodic patterns (0-1).
    Uses 3-gram interval patterns. Higher = more repetition.
    """
    notes = get_notes(midi)
    if len(notes) < 5:
        return 0.0
    
    pitches = [n[0] for n in notes]
    ngrams = _get_interval_ngrams(pitches, n=3)
    
    if not ngrams:
        return 0.0
    
    # Count occurrences
    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
    
    # Fraction that appear more than once
    repeated = sum(1 for ng in ngrams if counts[ng] > 1)
    return repeated / len(ngrams)

# Interval dissonance scores (semitones -> dissonance)
# Based on roughness/critical bandwidth theory
INTERVAL_DISSONANCE = {
    0: 0.0,   # unison
    1: 1.0,   # minor 2nd (most dissonant)
    2: 0.8,   # major 2nd
    3: 0.2,   # minor 3rd
    4: 0.2,   # major 3rd
    5: 0.1,   # perfect 4th
    6: 0.9,   # tritone
    7: 0.05,  # perfect 5th
    8: 0.25,  # minor 6th
    9: 0.25,  # major 6th
    10: 0.7,  # minor 7th
    11: 0.8,  # major 7th
}

def _chord_dissonance(pitches: list[int]) -> float:
    """Compute dissonance of a set of simultaneous pitches."""
    if len(pitches) < 2:
        return 0.0
    
    total = 0.0
    count = 0
    for i in range(len(pitches)):
        for j in range(i + 1, len(pitches)):
            interval = abs(pitches[i] - pitches[j]) % 12
            total += INTERVAL_DISSONANCE.get(interval, 0.5)
            count += 1
    
    return total / count if count > 0 else 0.0

@feature
def harmonic_tension_curve(midi: mido.MidiFile) -> list[float]:
    """
    Time series of harmonic tension (8 segments).
    Based on interval dissonance of simultaneous notes.
    """
    notes = get_notes(midi)
    if not notes:
        return [0.0] * 8
    
    duration = midi.length
    if duration == 0:
        return [0.0] * 8
    
    # Divide into 8 segments
    n_segments = 8
    segment_dur = duration / n_segments
    tension = []
    
    for seg in range(n_segments):
        seg_start = seg * segment_dur
        seg_end = (seg + 1) * segment_dur
        
        # Find notes active in this segment
        seg_notes = [n for n in notes if n[2] < seg_end and n[2] + n[3] > seg_start]
        
        if len(seg_notes) < 2:
            tension.append(0.0)
            continue
        
        # Sample a few timepoints and average dissonance
        samples = []
        for t in np.linspace(seg_start, seg_end, 5):
            active = [n[0] for n in seg_notes if n[2] <= t < n[2] + n[3]]
            if len(active) >= 2:
                samples.append(_chord_dissonance(active))
        
        tension.append(float(np.mean(samples)) if samples else 0.0)
    
    return tension

@feature
def polyrhythm_detected(midi: mido.MidiFile) -> bool:
    """
    Detect if multiple conflicting metric grids coexist.
    Checks for 3:2 or 4:3 polyrhythms using IOI analysis.
    """
    notes = get_notes(midi)
    if len(notes) < 20:
        return False
    
    # Get inter-onset intervals
    onsets = sorted(set(n[2] for n in notes))
    if len(onsets) < 10:
        return False
    
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets) - 1)]
    iois = [x for x in iois if x > 0.01]  # filter tiny gaps
    
    if len(iois) < 5:
        return False
    
    # Cluster IOIs and check for 3:2 or 4:3 ratios
    iois = np.array(iois)
    median_ioi = np.median(iois)
    
    # Normalize by median
    ratios = iois / median_ioi
    
    # Check for presence of both ~1.0 and ~1.5 (3:2) or ~1.0 and ~1.33 (4:3)
    near_1 = np.sum((ratios > 0.85) & (ratios < 1.15))
    near_1_5 = np.sum((ratios > 1.4) & (ratios < 1.6))
    near_1_33 = np.sum((ratios > 1.25) & (ratios < 1.42))
    near_0_67 = np.sum((ratios > 0.6) & (ratios < 0.75))
    near_0_75 = np.sum((ratios > 0.7) & (ratios < 0.82))
    
    total = len(ratios)
    threshold = 0.1 * total  # at least 10% of each
    
    # 3:2 polyrhythm
    if near_1 > threshold and (near_1_5 > threshold or near_0_67 > threshold):
        return True
    # 4:3 polyrhythm
    if near_1 > threshold and (near_1_33 > threshold or near_0_75 > threshold):
        return True
    
    return False

# --- END FEATURES ---

def extract(midi: mido.MidiFile) -> dict[str, float | list[float]]:
    """Run all registered features on a MIDI file."""
    return {name: fn(midi) for name, fn in FEATURES.items()}

def extract_array(midi: mido.MidiFile) -> np.ndarray:
    """Run all features, flatten to array."""
    results = extract(midi)
    flat = []
    for v in results.values():
        if isinstance(v, (list, np.ndarray)):
            flat.extend(v)
        else:
            flat.append(v)
    return np.array(flat, dtype=np.float32)

# CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        m = load_midi(sys.argv[1])
        print(extract(m))