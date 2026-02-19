# midi-rubrics

```
 ╔╦╗╦╔╦╗╦  ╦═╗╦ ╦╔╗ ╦═╗╦╔═╗╔═╗
 ║║║║ ║║║  ╠╦╝║ ║╠╩╗╠╦╝║║  ╚═╗
 ╩ ╩╩═╩╝╩  ╩╚═╚═╝╚═╝╩╚═╩╚═╝╚═╝
```

Feature extraction for MIDI. 50+ musical descriptors, no ML, one function call.

---

## quickstart

```python
from rubrics import load_midi, extract

midi = load_midi("track.mid")
features = extract(midi)
```

---

## features

### structure
| feature | description |
|---------|-------------|
| `note_count` | total notes |
| `duration_seconds` | track length |
| `track_count` | MIDI tracks |
| `silence_ratio` | fraction of silence |

### rhythm
| feature | description |
|---------|-------------|
| `tempo_bpm` | beats per minute |
| `note_density` | notes per second |
| `avg_note_duration` | mean note length |
| `note_duration_variance` | rhythmic consistency |

### pitch
| feature | description |
|---------|-------------|
| `pitch_range` | highest - lowest note |
| `avg_pitch` | mean MIDI note |
| `pitch_std` | pitch spread |
| `unique_pitches` | distinct notes used |
| `pitch_class_histogram` | 12-element distribution |
| `melodic_interval_avg` | mean interval size |
| `melodic_interval_std` | interval variance |
| `contour_direction` | ratio ascending (0-1) |
| `leap_ratio` | fraction > 4 semitones |
| `stepwise_motion` | fraction <= 2 semitones |

### harmony
| feature | description |
|---------|-------------|
| `simultaneous_notes_avg` | mean polyphony |
| `simultaneous_notes_max` | max voices |
| `detected_key` | Krumhansl-Schmuckler (0-23) |
| `key_clarity` | tonal strength (0-1) |
| `voice_independence` | contrapuntal measure |
| `harmonic_tension_curve` | 8-segment dissonance |
| `motif_recurrence` | pattern repetition |

### dynamics
| feature | description |
|---------|-------------|
| `avg_velocity` | mean loudness |
| `velocity_std` | dynamic variation |
| `velocity_range` | max - min velocity |
| `dynamic_contour` | slope over time |
| `accent_ratio` | fraction > 100 |

### texture
| feature | description |
|---------|-------------|
| `channel_count` | active MIDI channels |
| `program_changes` | instrument switches |
| `instruments_used` | GM program numbers |
| `bass_presence` | fraction < MIDI 48 |
| `treble_presence` | fraction > MIDI 72 |
| `register_spread` | pitch std |

### drums (channel 10)
| feature | description |
|---------|-------------|
| `kick_density` | kicks per bar |
| `snare_density` | snares per bar |
| `hihat_density` | hihats per bar |
| `kick_snare_ratio` | kick/snare balance |
| `hihat_open_closed_ratio` | open/closed balance |
| `ride_vs_hihat` | ride prominence |
| `crash_frequency` | crashes per bar |
| `tom_usage` | toms per bar |
| `percussion_layers` | simultaneous drums |
| `drum_elements_used` | distinct percussion |

### per-track
All of the following return `list[float]` with one value per track:

- `note_count_per_track`
- `note_density_per_track`
- `avg_pitch_per_track`
- `pitch_range_per_track`
- `avg_velocity_per_track`
- `avg_duration_per_track`
- `unique_pitches_per_track`
- `polyphony_per_track`

### detection
| feature | description |
|---------|-------------|
| `note_density_high` | density > 8 notes/sec |
| `chord_heavy` | polyphony > 3 |
| `monophonic` | polyphony < 1.2 |
| `polyrhythm_detected` | 3:2 or 4:3 patterns |

---

## key detection

Uses Krumhansl-Schmuckler algorithm. Correlates the pitch class histogram against major/minor profiles from Krumhansl & Kessler (1982).

Returns 0-11 for C-B major, 12-23 for C-B minor.

---

## harmonic tension

8-segment time series of harmonic dissonance. Uses interval roughness scores:

| interval | dissonance |
|----------|------------|
| unison, P5 | 0.0, 0.05 |
| m3, M3, P4 | 0.1-0.2 |
| m6, M6 | 0.25 |
| M2, M7 | 0.8 |
| tritone | 0.9 |
| m2 | 1.0 |

---

## api

| function | returns |
|----------|---------|
| `load_midi(path)` | `MidiFile` |
| `extract(midi)` | `dict[str, float \| list]` |
| `extract_array(midi)` | `np.ndarray` (flattened) |
| `get_notes(midi)` | `list[(pitch, vel, start, dur)]` |
| `get_drum_notes(midi)` | `list[(note, vel, start, dur)]` |

---

## install

```bash
pip install mido numpy rich
```

---

## test

```bash
echo "/path/to/midi/folder" > test_data.txt
python test_rubrics.py
# -> test_results.json
```

---

## dependencies

- `mido` - MIDI parsing
- `numpy` - numerical operations
- `rich` - terminal output (test script only)

---

*for music information retrieval research*
