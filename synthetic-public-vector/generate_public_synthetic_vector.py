#!/usr/bin/env python3
"""Construct a wholly synthetic DICOM object and a deterministic CFP9-v2 interop vector.

Research-only public key/nonce.  The original AMIE-BakerChen encryption,
authentication and native DICOM package serialization modules are imported
unchanged from the user's frozen source tree.  No patient data is accessed.

Usage: python generate_public_synthetic_vector.py --source-root /path/to/src --out /path/to/public-vector
If pydicom is unavailable in a restricted test environment, a *synthetic-only*
import shim can be enabled with AMIE_SYNTHETIC_IMPORT_SHIM=1.  The production
DICOM input reader is not used in this script: only an explicitly constructed
and validated 128x128 single-frame monochrome 16-bit synthetic raster is used.
"""
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys, types, uuid
from pathlib import Path

ROWS=128
COLS=128
SOP_CLASS='1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage syntax; image is FAKE
TS='1.2.840.10008.1.2.1'  # Explicit VR Little Endian
KEY=hashlib.sha256(b'AMIE-BakerChen|PUBLIC-SYNTHETIC-ONLY|CFP9-v2|master-key').digest()
NONCE=hashlib.sha256(b'AMIE-BakerChen|PUBLIC-SYNTHETIC-ONLY|CFP9-v2|nonce').digest()[:16]


def ui(text):
    b=text.encode('ascii'); return b + (b'\0' if len(b)%2 else b'')


def ascii_pad(text, pad=b' '):
    b=text.encode('ascii'); return b+(pad if len(b)%2 else b'')


def element(group, num, vr, value):
    if len(value)%2: raise ValueError('DICOM element value must have even length')
    head=struct.pack('<HH',group,num)+vr.encode('ascii')
    if vr in ('OB','OW','SQ','UN','UT'):
        return head+b'\0\0'+struct.pack('<I',len(value))+value
    return head+struct.pack('<H',len(value))+value


def uid(name):
    return '2.25.'+str(uuid.uuid5(uuid.NAMESPACE_URL,'AMIE-BakerChen synthetic DICOM fixture '+name).int)


def build_dicom():
    # Entirely synthetic identifiers and explicit NO_PATIENT strings.
    sop=uid('sop');study=uid('study');series=uid('series');implement=uid('implementation')
    meta_items=[
        element(0x0002,0x0001,'OB',b'\x00\x01'),
        element(0x0002,0x0002,'UI',ui(SOP_CLASS)),
        element(0x0002,0x0003,'UI',ui(sop)),
        element(0x0002,0x0010,'UI',ui(TS)),
        element(0x0002,0x0012,'UI',ui(implement)),
        element(0x0002,0x0013,'SH',ascii_pad('AMIE_SYNTH_V1')),
    ]
    meta = element(0x0002,0x0000,'UL',struct.pack('<I',sum(map(len,meta_items))))+b''.join(meta_items)
    v=[]
    for r in range(ROWS):
        for c in range(COLS):
            # Artificial integer pattern, not derived from any medical data.
            pixel=((r*17+c*37+((r-63)**2+(c-63)**2)//4+(r^c)*11)%4096)
            v.append(pixel)
    pixel_data=struct.pack('<'+'H'*len(v),*v)
    ds=[
        element(0x0008,0x0016,'UI',ui(SOP_CLASS)),
        element(0x0008,0x0018,'UI',ui(sop)),
        element(0x0008,0x0060,'CS',ascii_pad('MR')),
        element(0x0010,0x0010,'PN',ascii_pad('SYNTHETIC^NO_PATIENT')),
        element(0x0010,0x0020,'LO',ascii_pad('SYNTHETIC_ONLY')),
        element(0x0020,0x000d,'UI',ui(study)),
        element(0x0020,0x000e,'UI',ui(series)),
        element(0x0028,0x0002,'US',struct.pack('<H',1)),
        element(0x0028,0x0004,'CS',ascii_pad('MONOCHROME2')),
        element(0x0028,0x0010,'US',struct.pack('<H',ROWS)),
        element(0x0028,0x0011,'US',struct.pack('<H',COLS)),
        element(0x0028,0x0100,'US',struct.pack('<H',16)),
        element(0x0028,0x0101,'US',struct.pack('<H',12)),
        element(0x0028,0x0102,'US',struct.pack('<H',11)),
        element(0x0028,0x0103,'US',struct.pack('<H',0)),
        element(0x7fe0,0x0010,'OW',pixel_data),
        # Nonempty suffix also exercises the DICOM suffix authentication path.
        element(0x9999,0x0010,'LO',ascii_pad('SYNTHETIC_AFTER_PIXEL')),
    ]
    return b'\0'*128+b'DICM'+meta+b''.join(ds)


def setup_modules(src):
    sys.path.insert(0,str(src))
    # The full source imports pydicom for its general-purpose DICOM loaders.
    # Here we do NOT invoke those loaders. This fallback only lets the untouched
    # cryptographic/package modules import in offline CI lacking pydicom.
    try:
        import pydicom  # noqa: F401
    except ImportError:
        if os.environ.get('AMIE_SYNTHETIC_IMPORT_SHIM') != '1':
            raise SystemExit('Install pydicom to run, or set AMIE_SYNTHETIC_IMPORT_SHIM=1 for synthetic-only test (no patient DICOM).')
        p=types.ModuleType('pydicom');d=types.ModuleType('pydicom.dataset')
        d.Dataset=type('Dataset',(),{})
        p.dataset=d
        sys.modules['pydicom']=p;sys.modules['pydicom.dataset']=d
        print('NOTICE: synthetic-only import shim for absent pydicom; general DICOM reader NOT tested here.',file=sys.stderr)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True,help='Directory containing original CFP9-v2 Python modules')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    setup_modules(args.source_root)
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from dicom_native_io import split_native_dicom_bytes
    from dicom_native_adapter import NativePixelLayout,pixel_data_to_uint8_matrix
    from v4_secure_pipeline import encrypt_image_secure,ALGORITHM_VERSION,BLOCK_SIZE
    from native_dicom_authenticated_package_CFP9 import (
        AuthenticatedNativeDicomPackage,build_dicom_context,_derive_dicom_envelope_key,
        compute_dicom_envelope_tag,decrypt_native_dicom_package,DICOM_PROFILE
    )
    from native_dicom_package_serialization_CFP9 import (
        serialize_native_dicom_package,deserialize_native_dicom_package,
        canonical_json_bytes,EXPERIMENT_VERSION,PACKAGE_MAGIC,FIELD_ORDER
    )
    if ALGORITHM_VERSION!='AMIE-BakerChen-CFP9-v2': raise RuntimeError('wrong algorithm version')
    raw=build_dicom()
    parts=split_native_dicom_bytes(raw)
    if parts.pixel_data_length!=ROWS*COLS*2 or len(parts.suffix)==0:
        raise RuntimeError('synthetic DICOM split mismatch')
    layout=NativePixelLayout(
        rows=ROWS,columns=COLS,samples_per_pixel=1,number_of_frames=1,
        bits_allocated=16,bits_stored=12,high_bit=11,pixel_representation=0,
        photometric_interpretation='MONOCHROME2',transfer_syntax_uid=TS,
        bytes_per_sample=2,expected_pixel_data_bytes=ROWS*COLS*2,
        matrix_rows=ROWS,matrix_columns=COLS*2
    )
    native=pixel_data_to_uint8_matrix(parts.pixel_data,layout)
    with ProcessPoolExecutor(max_workers=2) as executor:
        enc=encrypt_image_secure(native,KEY,session_nonce=NONCE,block_size=BLOCK_SIZE,max_workers=2,executor=executor)
        pkg=AuthenticatedNativeDicomPackage(
            encrypted_image=np.asarray(enc['encrypted_image'],dtype=np.uint8),
            amie_metadata=enc['metadata'],amie_authentication_tag=enc['authentication_tag'],
            dicom_prefix=parts.prefix,dicom_suffix=parts.suffix,
            dicom_context=build_dicom_context(parts=parts,layout=layout,native_matrix=native),
            dicom_authentication_tag=b''
        )
        pkg.dicom_authentication_tag=compute_dicom_envelope_tag(pkg,_derive_dicom_envelope_key(KEY,NONCE))
        serialized=serialize_native_dicom_package(pkg)
        reparsed=deserialize_native_dicom_package(serialized)
        assert serialize_native_dicom_package(reparsed)==serialized
        recovered,ev=decrypt_native_dicom_package(reparsed,KEY,executor=executor)
    assert raw==recovered,'Python native DICOM round-trip must be byte-exact'
    sha=lambda b:hashlib.sha256(b).hexdigest()
    ctx=canonical_json_bytes(pkg.dicom_context); meta=canonical_json_bytes(pkg.amie_metadata)
    cipher=np.ascontiguousarray(pkg.encrypted_image).tobytes(order='C')
    vec={
        'test_vector_version':'AMIE-Native-DICOM-Interop-Vector-CFP9-v2',
        'synthetic_only':True,
        'profile':DICOM_PROFILE,
        'algorithm_version':ALGORITHM_VERSION,
        'purpose':'Entirely fabricated reproducibility fixture. NEVER clinical input. Key and nonce are PUBLIC TEST MATERIAL.',
        'validation_master_key_hex':KEY.hex(),
        'validation_key_is_public_test_vector_only':True,
        'validation_session_nonce_hex':NONCE.hex(),
        'production_nonce_rule':'Fresh, never-reused 128-bit session nonce required',
        'representation':'128x128, single-frame 16-bit allocated/12-bit stored unsigned MONOCHROME2',
        'serialization':{
            'magic_hex':PACKAGE_MAGIC.hex(),
            'field_count':len(FIELD_ORDER),
            'field_order':[x.decode('ascii') for x in FIELD_ORDER],
            'field_count_encoding':'uint16 big-endian',
            'field_framing':'uint16_be(label_length) || label || uint64_be(payload_length) || payload',
            'json_canonicalization':"UTF-8 JSON; sort_keys=True; separators=(',', ':'); ensure_ascii=True",
            'ciphertext_encoding':'uint8 row-major C-order bytes'
        },
        'expected_hashes':{
            'serialized_package_sha256':sha(serialized),
            'original_dicom_sha256':sha(raw),
            'recovered_dicom_sha256':sha(recovered),
            'dicom_prefix_sha256':sha(parts.prefix),
            'dicom_suffix_sha256':sha(parts.suffix),
            'dicom_context_canonical_json_sha256':sha(ctx),
            'amie_metadata_canonical_json_sha256':sha(meta),
            'amie_ciphertext_sha256':sha(cipher)
        },
        'expected_tags':{
            'amie_inner_hmac_sha256_hex':pkg.amie_authentication_tag.hex(),
            'dicom_outer_hmac_sha256_hex':pkg.dicom_authentication_tag.hex()
        },
        'expected_ciphertext_shape':list(map(int,pkg.encrypted_image.shape)),
        'expected_dicom_suffix_length':len(parts.suffix),
        'receiver_pass_condition':'Go verifies two HMAC tags, recovers DICOM with original_dicom_sha256, and outputs native_dicom_decrypted=true.'
    }
    out=args.out
    (out/'synthetic_128x128_original.dcm').write_bytes(raw)
    (out/'synthetic_128x128.amiedcm').write_bytes(serialized)
    (out/'synthetic_128x128_test_vector.json').write_text(json.dumps(vec,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PYTHON_PASS','dicom_sha256':sha(raw),'package_sha256':sha(serialized),'prefix_bytes':len(parts.prefix),'suffix_bytes':len(parts.suffix),'cipher_shape':list(pkg.encrypted_image.shape)}))

if __name__=='__main__':main()
