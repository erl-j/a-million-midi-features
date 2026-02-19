# MIDI Feature Ideas

Feasibility: **easy** (direct from MIDI), **medium** (derived/statistical), **hard** (requires music theory/ML)

## Rhythm & Timing

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `tempo_bpm` | Extract tempo from MIDI meta messages | easy | ✅ |
| `note_density` | Notes per second | easy | ✅ |
| `avg_note_duration` | Mean duration of all notes | easy | ✅ |
| `note_duration_variance` | How consistent are note lengths | easy | ✅ |
| `onset_regularity` | How evenly spaced are note onsets (low = rubato, high = quantized) | medium |
| `syncopation_score` | Ratio of notes on off-beats vs on-beats | medium |
| `swing_ratio` | Ratio of long/short in pairs (detects swing feel) | medium |
| `rhythmic_entropy` | Shannon entropy of inter-onset intervals (complexity) | medium |
| `tempo_stability` | Variance in local tempo estimates | medium |
| `polyrhythm_detected` | Whether multiple time signatures coexist | hard |

## Pitch & Melody

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `pitch_range` | Highest - lowest MIDI note | easy | ✅ |
| `avg_pitch` | Mean MIDI note number | easy | ✅ |
| `pitch_std` | Standard deviation of pitches | easy | ✅ |
| `unique_pitches` | Number of distinct pitches used | easy | ✅ |
| `pitch_class_histogram` | 12-element vector of pitch class frequencies | easy | ✅ |
| `melodic_interval_avg` | Mean interval size between consecutive notes | easy | ✅ |
| `melodic_interval_std` | Variance in melodic intervals | easy | ✅ |
| `contour_direction` | Ratio of ascending vs descending intervals | easy | ✅ |
| `leap_ratio` | Fraction of intervals > 4 semitones | easy | ✅ |
| `stepwise_motion` | Fraction of intervals <= 2 semitones | easy | ✅ |
| `pitch_entropy` | Shannon entropy of pitch distribution | medium |
| `melodic_complexity` | Lempel-Ziv complexity of pitch sequence | medium |

## Harmony & Chords

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `simultaneous_notes_avg` | Average polyphony (notes sounding together) | easy | ✅ |
| `simultaneous_notes_max` | Maximum polyphony | easy | ✅ |
| `chord_count` | Number of distinct vertical sonorities | medium |
| `consonance_score` | Ratio of consonant intervals (3rds, 5ths, octaves) | medium |
| `dissonance_score` | Ratio of dissonant intervals (2nds, 7ths, tritones) | medium |
| `seventh_chord_ratio` | How often 4+ notes sound (jazz indicator) | medium |
| `chord_change_rate` | Harmonic rhythm - chord changes per bar | medium |
| `key_clarity` | Krumhansl-Schmuckler key-finding correlation strength | hard |
| `detected_key` | Most likely key (0-23 for major/minor) | hard |
| `modal_mixture` | Presence of borrowed chords from parallel mode | hard |
| `harmonic_tension_curve` | Time series of harmonic tension | hard |

## Dynamics & Velocity

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `avg_velocity` | Mean note velocity | easy | ✅ |
| `velocity_std` | Velocity variance (expressive range) | easy | ✅ |
| `velocity_range` | Max - min velocity | easy | ✅ |
| `dynamic_contour` | Whether piece gets louder/softer over time | easy | ✅ |
| `accent_ratio` | Ratio of notes with velocity > 100 | easy | ✅ |
| `velocity_entropy` | Entropy of velocity distribution | medium |
| `humanization_score` | Micro-timing/velocity variance (human vs quantized) | medium |

## Structure & Form

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `track_count` | Number of MIDI tracks | easy | ✅ |
| `total_duration` | Length in seconds | easy | ✅ |
| `note_count` | Total notes | easy | ✅ |
| `silence_ratio` | Fraction of time with no notes | easy | ✅ |
| `phrase_count` | Number of detected phrases (gaps > threshold) | medium |
| `avg_phrase_length` | Mean phrase duration | medium |
| `repetition_score` | Self-similarity of pitch patterns | medium |
| `motif_recurrence` | Frequency of repeated short patterns | hard |
| `structural_segments` | Detected sections (intro/verse/chorus) | hard |

## Texture & Orchestration

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `channel_count` | Number of active MIDI channels | easy | ✅ |
| `program_changes` | Number of instrument changes | easy | ✅ |
| `instruments_used` | List of GM program numbers | easy | ✅ |
| `bass_presence` | Notes in low register (< MIDI 48) | easy | ✅ |
| `treble_presence` | Notes in high register (> MIDI 72) | easy | ✅ |
| `register_spread` | How spread across registers | easy | ✅ |
| `voice_independence` | Correlation between tracks (low = counterpoint) | hard |

## Genre/Style Indicators

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `note_density_high` | > 8 notes/sec (suggests fast music) | easy | ✅ |
| `chord_heavy` | Avg polyphony > 3 (suggests pads/chords) | easy | ✅ |
| `monophonic` | Avg polyphony ~1 (melody-focused) | easy | ✅ |
| `quantization_level` | How grid-aligned (electronic vs human) | medium |
| `blues_scale_affinity` | Pitch alignment with blues scale | medium |
| `jazz_voicing_score` | Extended chord tones (9ths, 11ths, 13ths) | hard |

---

## Drums & Percussion

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| `kick_density` | Kick drum (note 36) hits per bar | easy | ✅ |
| `snare_density` | Snare (note 38/40) hits per bar | easy | ✅ |
| `hihat_density` | Hi-hat (notes 42/44/46) hits per bar | easy | ✅ |
| `kick_snare_ratio` | Balance between kick and snare | easy | ✅ |
| `drum_elements_used` | Count of distinct percussion notes | easy | ✅ |
| `hihat_open_closed_ratio` | Open (46) vs closed (42) hi-hats | easy | ✅ |
| `backbeat_strength` | Snare emphasis on beats 2 and 4 | medium |
| `downbeat_kick_ratio` | Kicks on beat 1 of each bar | medium |
| `four_on_floor` | Kick on every quarter note (house/techno detector) | medium |
| `breakbeat_score` | Syncopation level in kick pattern | medium |
| `ghost_note_ratio` | Low velocity hits (< 60) in snare/hi-hat | medium |
| `hihat_pattern_entropy` | Complexity of hi-hat rhythm | medium |
| `kick_pattern_entropy` | Complexity of kick rhythm | medium |
| `drum_velocity_groove` | Velocity variation across the bar (swing feel) | medium |
| `fill_frequency` | Detected fills/breaks per 8 bars | medium |
| `ride_vs_hihat` | Ratio of ride (51/59) to hi-hat usage | easy | ✅ |
| `crash_frequency` | Crash cymbal (49/57) hits per section | easy | ✅ |
| `tom_usage` | Presence/frequency of tom fills | easy | ✅ |
| `percussion_layers` | Simultaneous percussion voices | easy | ✅ |
| `groove_template_match` | Similarity to common patterns (boom-bap, trap, etc.) | hard |
| `microtiming_swing` | Deviation from grid on offbeats | medium |
| `trap_hihat_rolls` | Detected 32nd/64th hi-hat sequences | medium |
| `polyrhythm_percussion` | Conflicting rhythmic cycles (3 vs 4) | hard |

---

## Priority Recommendations

Start with these (high value, easy to implement):
1. `pitch_class_histogram` - foundational for harmony analysis
2. `note_density` - instant genre signal
3. `avg_velocity` + `velocity_std` - dynamics info
4. `simultaneous_notes_avg` - texture indicator
5. `melodic_interval_avg` + `leap_ratio` - melodic character
6. `onset_regularity` - rhythm quantization level

Second wave (medium effort, high insight):
1. `rhythmic_entropy` - rhythmic complexity
2. `consonance_score` - harmonic character
3. `repetition_score` - structural insight
4. `humanization_score` - production style

