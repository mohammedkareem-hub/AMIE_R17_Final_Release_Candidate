#!/usr/bin/env python3
"""Immutable snapshot checksum + synthetic-only native DICOM / Go interoperability tests.

Does NOT verify patient data, source permissions, formal security or clinical suitability.
"""
from __future__ import annotations
import hashlib, subprocess, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(argv,cwd=ROOT):
 print('RUN', ' '.join(map(str,argv)),flush=True)
 subprocess.run(argv,cwd=cwd,check=True,timeout=240)

def main():
 manifest=ROOT/'SHA256SUMS.txt'
 count=0
 for entry in manifest.read_text(encoding='utf-8').splitlines():
  if not entry:continue
  digest,rel=entry.split('  ',1)
  p=ROOT/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
   raise AssertionError('source inventory mismatch: '+rel)
  count+=1
 print('PASS: staging hashes',count)
 if not shutil.which('go'):
  raise SystemExit('FAIL: Go >=1.23 required for independent check')
 run([sys.executable,'synthetic-public-vector/verify_public_synthetic_vector.py','--directory','synthetic-public-vector'])
 run(['go','test','./...'],ROOT/'go-receiver')
 run(['go','run','.','-package','../synthetic-public-vector/synthetic_128x128.amiedcm', '-vector','../synthetic-public-vector/synthetic_128x128_test_vector.json'],ROOT/'go-receiver')
 print('PASS: synthetic framing and independent Go recovery; NOT a medical DICOM rerun')
if __name__=='__main__': main()
