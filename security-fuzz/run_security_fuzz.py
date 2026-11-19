#!/usr/bin/env python3
"""AMIE CFP9-v2 Phase 7: bounded, reproducible native-package robustness tests.

Run ONLY on an authorized PUBLIC research test vector, never on hospital data.
The program never writes the test vector, recovered DICOM, or keys to results.
All mutated packages reside inside a TemporaryDirectory.
Does not modify the CFP9 encryption implementation, nor the frozen results.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import struct
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

MAGIC = b'AMIE-NATIVE-DICOM-PACKAGE\x00\x01'
LABELS = [b'dicom-prefix', b'dicom-suffix', b'dicom-context', b'amie-metadata',
          b'amie-ciphertext', b'amie-inner-tag', b'dicom-outer-tag']
PREFIX = b'MedicalImageEncryption-v1|'
MAX_BYTES = 32 * 1024 * 1024


def frame(label: bytes, payload: bytes) -> bytes:
    return struct.pack('>H',len(label)) + label + struct.pack('>Q',len(payload)) + payload


def parse_offsets(blob: bytes):
    if not blob.startswith(MAGIC):
        raise ValueError('fixture magic mismatch')
    p = len(MAGIC)
    n = struct.unpack_from('>H', blob, p)[0]; p += 2
    if n != 7: raise ValueError('expected seven frames')
    frames = []
    for want in LABELS:
        label_start = p
        ll = struct.unpack_from('>H', blob, p)[0]; p += 2
        label = blob[p:p+ll]; p += ll
        if label != want: raise ValueError('fixture frame label mismatch')
        len_at = p
        size = struct.unpack_from('>Q', blob, p)[0]; p += 8
        start = p
        p += size
        if p > len(blob): raise ValueError('fixture truncated')
        frames.append({'label':want, 'label_at':label_start, 'len_at':len_at,
                       'start':start, 'end':p, 'payload':blob[start:p]})
    if p != len(blob): raise ValueError('fixture unexpected trailing bytes')
    return frames


def serialize(payloads: list[bytes]) -> bytes:
    return MAGIC + struct.pack('>H',7) + b''.join(frame(label,data) for label,data in zip(LABELS,payloads))


def canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()


def key_derive(master: bytes, nonce: bytes, context: bytes) -> bytes:
    extract = hmac.digest(nonce,master,'sha256')
    return hmac.digest(extract, PREFIX+context+b'\x01','sha256')


def authorized_repackage(frames, master: bytes, *, metadata=None, context=None, ciphertext=None):
    payloads = [f['payload'] for f in frames]
    if metadata is not None: payloads[3] = canon(metadata)
    if context is not None: payloads[2] = canon(context)
    if ciphertext is not None: payloads[4] = ciphertext
    meta = json.loads(payloads[3])
    nonce = bytes.fromhex(meta['session_nonce_hex'])
    inner_key = key_derive(master, nonce, b'ciphertext-authentication-key')
    outer_key = key_derive(master, nonce, b'native-dicom-envelope-authentication-key')
    payloads[5] = hmac.digest(inner_key,b'Ciphertext-v1|'+payloads[3]+b'|'+payloads[4], 'sha256')
    outer_message = b'AMIE-Native-DICOM-Envelope-v1|' + b''.join(frame(a,b) for a,b in zip(LABELS[:6],payloads[:6]))
    payloads[6] = hmac.digest(outer_key,outer_message,'sha256')
    return serialize(payloads)


def produce_cases(raw, frames, rng, master, base_count):
    # Return generator (category, detail, bytes, expected_reject, signed).
    for i in range(base_count):
        b = bytearray(raw); idx=rng.randrange(len(b)); b[idx] ^= 1<<rng.randrange(8)
        yield ('bit_flip', str(idx), bytes(b), True, False)
    for i in range(base_count//2):
        b = bytearray(raw); idx=rng.randrange(len(b)); b[idx] = (b[idx]+rng.randrange(1,256))%256
        yield ('byte_overwrite',str(idx),bytes(b),True,False)
    for i in range(120):
        cut = rng.randrange(0,len(raw))
        yield ('truncation',str(cut),raw[:cut],True,False)
    for i in range(120):
        extra=rng.randbytes(rng.randrange(1,48))
        yield ('trailing_bytes',str(len(extra)),raw+extra,True,False)
    for i in range(120):
        item=rng.choice(frames)
        idx=item['len_at']; b=bytearray(raw)
        b[idx:idx+8]=struct.pack('>Q', rng.choice([0,1,2,65535,len(raw),len(raw)*100,2**64-1]))
        if bytes(b)!=raw:
            yield ('field_length',item['label'].decode(),bytes(b),True,False)
    for i in range(100):
        item=rng.choice(frames)
        b=bytearray(raw); at=item['label_at']+2+rng.randrange(len(item['label'])); b[at] ^= 1
        yield ('frame_label',item['label'].decode(),bytes(b),True,False)
    for i in range(80):
        item=rng.choice([frames[2],frames[3]])
        b=bytearray(raw); idx=rng.randrange(item['start'],item['end']); b[idx] = rng.choice([0,255,34,92,123,125])
        if bytes(b)!=raw: yield ('json_framing',item['label'].decode(),bytes(b),True,False)

    # The following have VALID inner + outer HMACs signed with the PUBLIC test key.
    # They are not simulations of an unauthenticated network attacker. These
    # probe the receiver's behavior for malformed data from an authorized signer.
    ctx = json.loads(frames[2]['payload'])
    meta = json.loads(frames[3]['payload'])
    # Deliberate resource-cap probe (33 MiB): Go must reject before parsing.
    yield ('oversize_cap','32MiB_plus_one',b'X'*(MAX_BYTES+1),True,False)
    semantic_ctx = {
        'invalid_frame_count':('number_of_frames',2),
        'zero_rows':('rows',0),
        'wrong_columns':('columns',ctx['columns']+1),
        'incorrect_bits_stored':('bits_stored',0),
        'incorrect_high_bit':('high_bit',15),
        'invalid_pixel_representation':('pixel_representation',4),
        'incorrect_bytes_per_sample':('bytes_per_sample',1),
        'wrong_native_rows':('native_matrix_rows',ctx['native_matrix_rows']+1),
        'wrong_native_columns':('native_matrix_columns',ctx['native_matrix_columns']+1),
        'wrong_pixel_size':('pixel_data_length_bytes',ctx['pixel_data_length_bytes']+1),
        'wrong_prefix_hash':('dicom_prefix_sha256','0'*64),
        'wrong_prefix_length':('dicom_prefix_length_bytes',ctx['dicom_prefix_length_bytes']+2),
        'unsupported_dtype':('native_matrix_dtype','uint16'),
        'unsupported_policy':('pixel_data_policy','plaintext'),
        'negative_bits':('bits_allocated',-8),
        'invalid_native_shape_type':('native_matrix_columns','NaN'),
        'extreme_dimensions':('columns', 9223372036854775807),
    }
    for name,(field,value) in semantic_ctx.items():
        c=dict(ctx);c[field]=value
        yield ('authorized_invalid_context',name,authorized_repackage(frames,master,context=c),True,True)
    semantic_meta = {
        'negative_block_size':('block_size',-1),
        'zero_block_size':('block_size',0),
        'invalid_dtype':('dtype','uint16'),
        'too_large_shape':('padded_shape',[100000000,100000000]),
        'invalid_original_shape':('original_shape',[1,1]),
        'invalid_nonce_format':('session_nonce_hex','a'),
        'bad_context':('plaintext_context_hex','c'),
        'bad_partition_empty':('baker_partition',[]),
        'bad_partition_non_divisor':('baker_partition',[63,1]),
        'bad_partition_overfill':('baker_partition',[64,64]),
        'bad_partition_extra_blocks':('baker_partition',[32,32,32]),
    }
    for name,(field,value) in semantic_meta.items():
        m=dict(meta);m[field]=value
        if field=='session_nonce_hex':
            # Cannot derive keys for malformed nonce; mutate original authenticated metadata without re-signing.
            continue
        yield ('authorized_invalid_metadata',name,authorized_repackage(frames,master,metadata=m),True,True)


def check_one(go_binary:Path, file:Path, master:bytes, raw:bytes, *, timeout:float, decrypt:bool, expected_sha256:str|None=None):
    file.write_bytes(raw)
    cmd=[str(go_binary),'-package',str(file),'-key-hex',master.hex(),'-max-bytes',str(MAX_BYTES)]
    if not decrypt: cmd.append('-auth-only')
    if expected_sha256 is not None: cmd.extend(['-expected-sha256',expected_sha256])
    try:
        completed=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
        if completed.returncode == 0:
            return 'accepted',completed.stderr.decode(errors='replace')[:140]
        err=completed.stderr.decode(errors='replace')
        if 'panic:' in err or 'fatal error:' in err:
            return 'crash',err[:240]
        return 'rejected',err[:140]
    except subprocess.TimeoutExpired:
        return 'timeout','receiver exceeded timeout'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--go-bin',required=True,type=Path)
    p.add_argument('--package',required=True,type=Path)
    p.add_argument('--vector',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--random-bitflips',type=int,default=750)
    p.add_argument('--timeout',type=float,default=3.0)
    a=p.parse_args()
    start=time.perf_counter()
    rng=random.Random(a.seed)
    valid=a.package.read_bytes()
    frames=parse_offsets(valid)
    vector=json.loads(a.vector.read_text())
    key=bytes.fromhex(vector['validation_master_key_hex'])
    if len(key)!=32: raise RuntimeError('test key invalid')
    if hashlib.sha256(valid).hexdigest()!=vector['expected_hashes']['serialized_package_sha256']:
        raise RuntimeError('fixture SHA-256 did not match frozen vector')
    output={'experiment':'Phase7-Security-Fuzz-and-Robustness-CFP9-v2',
            'seed':a.seed,'reference_package_sha256':hashlib.sha256(valid).hexdigest(),
            'reference_dicom_sha256':vector['expected_hashes']['original_dicom_sha256'],
            'public_test_key_only':True,'source_algorithm_unchanged':True,
            'cases':0,'by_category':{},'findings':[], 'limit_sec_per_case':a.timeout}
    totals=Counter()
    failures=[]
    with tempfile.TemporaryDirectory(prefix='amie-fuzz-phase7-') as root:
        path=Path(root)/'trial.amiedcm'
        ok,note=check_one(a.go_bin.resolve(),path,key,valid,timeout=a.timeout,decrypt=True,expected_sha256=vector['expected_hashes']['original_dicom_sha256'])
        if ok!='accepted': raise RuntimeError(f'Valid signed frozen package failed full receiver: {ok}, {note}')
        output['valid_reference_full_recovery']='PASS'
        for category,detail,mutant,expected_reject,signed in produce_cases(valid,frames,rng,key,a.random_bitflips):
            actual,note=check_one(a.go_bin.resolve(),path,key,mutant,timeout=a.timeout,decrypt=signed)
            totals[(category,actual)]+=1
            output['cases']+=1
            if actual!='rejected':
                failures.append({'category':category,'case':detail,'observed':actual,
                                 'note':note,'signed_by_public_test_key':signed})
    summary={}
    for (category,outcome),value in sorted(totals.items()):
        summary.setdefault(category,{})[outcome]=value
    output.update({'by_category':summary,'findings':failures,'findings_count':len(failures),
                   'seconds_elapsed':round(time.perf_counter()-start,3),
                   'result':'PASS' if not failures else 'FINDINGS_REQUIRING_REMEDIATION',
                   'scope':'Mutational fuzzing of a single public test vector; not a formal security proof.'})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'result':output['result'],'cases':output['cases'],
                      'findings_count':len(failures),'by_category':summary,
                      'findings':failures,'seconds_elapsed':output['seconds_elapsed']},indent=2))
    if failures:
        raise SystemExit(2)

if __name__=='__main__':
    main()
