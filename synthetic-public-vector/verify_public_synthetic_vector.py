#!/usr/bin/env python3
"""Standard-library checks for the PUBLIC, entirely synthetic CFP9-v2 vector.
Does not decrypt and is not a substitute for running the independent Go receiver.
"""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path

MAGIC=b'AMIE-NATIVE-DICOM-PACKAGE\x00\x01'
LABELS=(b'dicom-prefix',b'dicom-suffix',b'dicom-context',b'amie-metadata',b'amie-ciphertext',b'amie-inner-tag',b'dicom-outer-tag')
sha=lambda x:hashlib.sha256(x).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    src=args.directory
    vector=json.loads((src/'synthetic_128x128_test_vector.json').read_text(encoding='utf-8'))
    dicom=(src/'synthetic_128x128_original.dcm').read_bytes()
    data=(src/'synthetic_128x128.amiedcm').read_bytes()
    assert vector['synthetic_only'] is True
    assert sha(data)==vector['expected_hashes']['serialized_package_sha256']
    assert sha(dicom)==vector['expected_hashes']['original_dicom_sha256']
    assert dicom[128:132]==b'DICM'
    assert data[:len(MAGIC)]==MAGIC
    pos=len(MAGIC)
    count=struct.unpack('>H',data[pos:pos+2])[0];pos+=2
    assert count==7
    fields={}
    for label in LABELS:
        length=struct.unpack('>H',data[pos:pos+2])[0];pos+=2
        actual=data[pos:pos+length];pos+=length
        assert actual==label
        plen=struct.unpack('>Q',data[pos:pos+8])[0];pos+=8
        payload=data[pos:pos+plen];pos+=plen
        assert len(payload)==plen
        fields[label.decode('ascii')]=payload
    assert pos==len(data)
    assert len(fields['dicom-suffix'])>0
    assert dicom==fields['dicom-prefix']+dicom[len(fields['dicom-prefix']):len(dicom)-len(fields['dicom-suffix'])]+fields['dicom-suffix']
    checks={
     'dicom_prefix_sha256': ('dicom-prefix','dicom_prefix_sha256'),
     'dicom_suffix_sha256': ('dicom-suffix','dicom_suffix_sha256'),
     'dicom_context_canonical_json_sha256': ('dicom-context','dicom_context_canonical_json_sha256'),
     'amie_metadata_canonical_json_sha256': ('amie-metadata','amie_metadata_canonical_json_sha256'),
     'amie_ciphertext_sha256': ('amie-ciphertext','amie_ciphertext_sha256')
    }
    for _,(label,key) in checks.items():
        assert sha(fields[label])==vector['expected_hashes'][key]
    assert fields['amie-inner-tag'].hex()==vector['expected_tags']['amie_inner_hmac_sha256_hex']
    assert fields['dicom-outer-tag'].hex()==vector['expected_tags']['dicom_outer_hmac_sha256_hex']
    for label in ('dicom-context','amie-metadata'):
        p=fields[label];obj=json.loads(p)
        assert json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()==p
    ctx=json.loads(fields['dicom-context'])
    assert ctx['rows']==128 and ctx['columns']==128
    assert ctx['bits_allocated']==16 and ctx['bits_stored']==12
    assert ctx['dicom_prefix_length_bytes']==len(fields['dicom-prefix'])
    assert ctx['dicom_suffix_length_bytes']==len(fields['dicom-suffix'])
    assert ctx['pixel_data_length_bytes']==128*128*2
    assert len(fields['amie-ciphertext'])==128*256
    assert b'SYNTHETIC' in dicom
    assert b'LIDC-IDRI' not in dicom and b'PatientName' not in dicom
    print(json.dumps({'pass':True,'synthetic_dicom_sha256':sha(dicom),'package_sha256':sha(data),'nonempty_suffix_bytes':len(fields['dicom-suffix']),'framing':'PASS','canonical_json':'PASS'}))

if __name__=='__main__': main()
