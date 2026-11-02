"""Verify package checksums, including unchanged saved result snapshots."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for item in manifest['files']:
    p=root/item['path'];assert p.is_file(),item['path']
    with p.open('rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==item['sha256'],item['path']
assert not any(p.suffix in {'.c','.so','.dylib'} for p in root.rglob('*'))
print('Package files and original result snapshots verified. No custom native code.')
