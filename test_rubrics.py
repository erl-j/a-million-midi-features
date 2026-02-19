"""Test script: load MIDI files from paths in test_data.txt and compute rubrics."""
from pathlib import Path
import json
import random
from rubrics import load_midi, extract
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.style import Style

# custom theme - warm gold accents on muted base
custom_theme = Theme({
    "info": "dim white",
    "accent": "#D4A03C",
    "accent.dim": "#A67C2E", 
    "success": "#7FB069",
    "error": "#C45B5B",
    "muted": "#6B7280",
    "bar.complete": "#D4A03C",
    "bar.finished": "#7FB069",
})

console = Console(theme=custom_theme)

LOGO = """
 ┌─────────────────────────────────────┐
 │                                     │
 │   ╔╦╗╦╔╦╗╦  ╦═╗╦ ╦╔╗ ╦═╗╦╔═╗╔═╗     │
 │   ║║║║ ║║║  ╠╦╝║ ║╠╩╗╠╦╝║║  ╚═╗     │
 │   ╩ ╩╩═╩╝╩  ╩╚═╚═╝╚═╝╩╚═╩╚═╝╚═╝     │
 │                                     │
 │   feature extraction for MIDI       │
 └─────────────────────────────────────┘
"""


def main():
    console.print(LOGO, style="accent", highlight=False)
    
    # read test paths
    test_data = Path("test_data.txt")
    paths = [line.strip() for line in test_data.read_text().splitlines() if line.strip()]
    
    results = {}
    errors = []
    
    for base_path in paths:
        base = Path(base_path)
        if not base.exists():
            console.print(f"  [error]✗[/error] path not found: [muted]{base}[/muted]")
            continue
        
        console.print(f"\n  [muted]scanning[/muted] [info]{base}[/info]")
        
        # collect files with live counter
        midi_files = []
        with console.status("[muted]indexing...[/muted]", spinner="line") as status:
            for pattern in ["*.mid", "*.MID", "*.midi"]:
                for f in base.rglob(pattern):
                    midi_files.append(f)
                    if len(midi_files) % 100 == 0:
                        status.update(f"[muted]{len(midi_files):,} files indexed[/muted]")
        
        random.shuffle(midi_files)
        midi_files = midi_files[:100]
        console.print(f"  [success]✓[/success] {len(midi_files):,} files\n")
        
        with Progress(
            SpinnerColumn(spinner_name="line", style="accent"),
            TextColumn("[muted]{task.description}[/muted]", justify="left"),
            BarColumn(bar_width=50, style="muted", complete_style="accent", finished_style="success"),
            TextColumn("[accent]{task.percentage:>5.1f}%[/accent]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("starting", total=len(midi_files))
            
            for midi_path in midi_files:
                # truncate filename elegantly
                name = midi_path.stem
                if len(name) > 45:
                    name = name[:42] + "..."
                progress.update(task, description=name)
                try:
                    midi = load_midi(midi_path)
                    features = extract(midi)
                    results[str(midi_path)] = features
                except Exception as e:
                    errors.append((str(midi_path), str(e)))
                progress.advance(task)
    
    console.print()
    
    # summary table - minimal borders
    table = Table(
        show_header=True,
        header_style="accent",
        border_style="muted",
        box=None,
        padding=(0, 2),
        collapse_padding=True,
    )
    table.add_column("", style="muted", width=20, justify="right")
    table.add_column("", style="info", justify="left")
    
    table.add_row("processed", f"{len(results):,}")
    table.add_row("errors", f"[error]{len(errors)}[/error]" if errors else "[success]0[/success]")
    table.add_row("features", str(len(list(results.values())[0])) if results else "0")
    
    console.print(Panel(
        table,
        title="[accent]summary[/accent]",
        border_style="muted",
        padding=(1, 2),
    ))
    
    # save results
    out_path = Path("test_results.json")
    
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return obj
    
    serializable = {k: {fk: convert(fv) for fk, fv in v.items()} for k, v in results.items()}
    out_path.write_text(json.dumps(serializable, indent=2))
    console.print(f"\n  [success]→[/success] {out_path}")
    
    if errors:
        console.print(f"\n  [error]errors ({len(errors)}):[/error]")
        for path, err in errors[:5]:
            console.print(f"    [muted]{Path(path).name}[/muted]")
            console.print(f"      [error]{err}[/error]")
        if len(errors) > 5:
            console.print(f"    [muted]+ {len(errors) - 5} more[/muted]")


if __name__ == "__main__":
    main()
