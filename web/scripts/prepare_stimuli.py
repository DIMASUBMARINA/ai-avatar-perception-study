#!/usr/bin/env python3
"""Prepare stimuli: identical processing for both classes, balanced 1:1, reproducible.

    python scripts/prepare_stimuli.py --real path/to/real --ai path/to/ai --generator stylegan3

Steps: pick N images per class (N = size of the smaller class unless --n given; real is sampled with a fixed seed),
centre-crop to a square, resize to --size, re-encode with the SAME format/quality (this also strips EXIF/metadata),
write to stimuli/real and stimuli/ai/<generator>, and save stimuli/manifest.csv (original file -> new file).
Needs: pip install pillow
"""
import argparse, csv, random, shutil, sys
from pathlib import Path
from PIL import Image, ImageOps

EXT = {'.jpg', '.jpeg', '.png', '.webp'}
ROOT = Path(__file__).resolve().parent.parent

def list_images(d):
    return sorted(p for p in Path(d).rglob('*') if p.suffix.lower() in EXT)

def process(src, dst, size, fmt, quality):
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.LANCZOS)
    if fmt == 'png':
        im.save(dst, 'PNG')
    else:
        im.save(dst, 'JPEG', quality=quality, subsampling=0, optimize=True)
    return (w, h)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--real', required=True); a.add_argument('--ai', required=True)
    a.add_argument('--generator', default='stylegan3', help='subfolder name under stimuli/ai (goes to the `generator` column)')
    a.add_argument('--n', type=int, help='images per class (default: size of the smaller class)')
    a.add_argument('--size', type=int, default=512); a.add_argument('--format', choices=['jpg', 'png'], default='jpg')
    a.add_argument('--quality', type=int, default=92); a.add_argument('--seed', type=int, default=42)
    a.add_argument('--out', default=str(ROOT / 'stimuli')); a.add_argument('--keep-demo', action='store_true')
    o = a.parse_args()

    real, ai = list_images(o.real), list_images(o.ai)
    n = o.n or min(len(real), len(ai))
    if len(real) < n or len(ai) < n:
        sys.exit(f'Not enough images: real={len(real)}, ai={len(ai)}, need {n}')
    rng = random.Random(o.seed)
    real_pick = sorted(rng.sample(real, n)); ai_pick = sorted(rng.sample(ai, n)) if len(ai) > n else ai
    print(f'found real={len(real)} ai={len(ai)} -> using {n} per class (1:1), seed={o.seed}')

    out = Path(o.out)
    if not o.keep_demo:
        for f in out.rglob('demo_*'): f.unlink()
    for d in (out / 'real', out / 'ai' / o.generator):
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)
    ext = 'jpg' if o.format == 'jpg' else 'png'
    rows, sizes = [], {'real': set(), 'ai': set()}
    for cls, picks, dest in (('real', real_pick, out / 'real'), ('ai', ai_pick, out / 'ai' / o.generator)):
        for i, src in enumerate(picks, 1):
            name = f'{"r" if cls == "real" else "a"}_{i:04d}.{ext}'
            sizes[cls].add(process(src, dest / name, o.size, o.format, o.quality))
            rows.append([cls, str(src), str(dest.relative_to(out) / name)])
    with open(out / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([['class', 'original_file', 'stimulus_file']] + rows)
    for cls in sizes:
        print(f'{cls}: source sizes seen: {sorted(sizes[cls])[:5]}{" ..." if len(sizes[cls]) > 5 else ""}')
    print(f'done: {2 * n} images at {o.size}x{o.size} {ext}; manifest -> {out / "manifest.csv"}')

if __name__ == '__main__':
    main()
