# -*- mode: python -*-
import os
import site
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# Need to specially handle these snowflakes
webrtcvad_path = None

for site_dir in site.getsitepackages():
    site_dir = Path(site_dir)
    webrtcvad_paths = list(site_dir.glob("_webrtcvad.*.so"))
    if webrtcvad_paths:
        webrtcvad_path = webrtcvad_paths[0]
        break

assert webrtcvad_path, "Missing webrtcvad"

a = Analysis(
    [Path.cwd() / "__main__.py"],
    pathex=["."],
    binaries=[(webrtcvad_path, ".")],
    datas=copy_metadata("webrtcvad"),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rhasspyasr_kaldi_hermes",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="rhasspyasr_kaldi_hermes",
)
