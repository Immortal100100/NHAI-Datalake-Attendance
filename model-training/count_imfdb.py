from pathlib import Path
d = Path('data/raw/imfdb/IMFDB FR dataset/IMFDB FR dataset')
actors = [x for x in d.iterdir() if x.is_dir()]
total = 0
counts = []
for x in actors:
    imgs = list(x.rglob('*.jpg')) + list(x.rglob('*.jpeg')) + list(x.rglob('*.png'))
    counts.append((len(imgs), x.name))
    total += len(imgs)
print(f'Actors: {len(actors)}, Total images: {total}')
for n, name in sorted(counts, reverse=True)[:10]:
    print(f'  {name}: {n}')
