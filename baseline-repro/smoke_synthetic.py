"""Synthetic-only reversibility checks. Not a security proof or clinical test."""
from __future__ import annotations
import hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from kamal2021_repro import Kamal2021
from afzal_p21 import AfzalP21Config,encrypt as afzal_encrypt,decrypt as afzal_decrypt
from liu2026_repro import Liu2026,Liu2026Config

def inputs():
    return {
        'ramp':np.arange(256,dtype=np.uint8).reshape(16,16),
        'constant':np.full((16,16),17,dtype=np.uint8),
        'checker':((np.indices((16,16)).sum(axis=0)%2)*255).astype(np.uint8),
        'non_square':np.arange(17*19,dtype=np.uint16).reshape(17,19).astype(np.uint8),
    }
def main():
    kamal=Kamal2021(block_size=16,a=3.9,n0=1000,rng_seed=2021)
    afzal=AfzalP21Config(alpha=3.99,x0=.123456789)
    for name,img in inputs().items():
        k,meta,_=kamal.encrypt_array(img)
        recovered,_=kamal.decrypt_array(k,meta)
        assert np.array_equal(img,recovered),(name,'Kamal')
        k2,_,_=kamal.encrypt_array(img)
        assert np.array_equal(k,k2),(name,'Kamal determinism')
        a=afzal_encrypt(img,seed='synthetic-non-secret-test-only',config=afzal)
        assert np.array_equal(img,afzal_decrypt(a,seed='synthetic-non-secret-test-only',config=afzal)),(name,'Afzal')
        a2=afzal_encrypt(img,seed='synthetic-non-secret-test-only',config=afzal)
        assert np.array_equal(a.image,a2.image),(name,'Afzal determinism')
    liu=Liu2026(Liu2026Config(it_rounds=3,stream_mode='fixed64_low32'))
    for name in ['ramp','checker']:
        img=inputs()[name]
        rgb=np.repeat(img[:,:,None],3,axis=2)
        c,metadata,_=liu.encrypt_rgb(rgb)
        p,_=liu.decrypt_rgb(c,metadata)
        assert np.array_equal(p,rgb),(name,'Liu')
        c2,_,_=liu.encrypt_rgb(rgb)
        assert np.array_equal(c,c2),(name,'Liu determinism')
    print('PASS: Kamal 4/4 Afzal 4/4 Liu 2/2 synthetic roundtrips + repeated deterministic encryption')
    print('SCOPE: synthetic 8-bit arrays; not original-author ciphertext equivalence or cross-OS byte identity')
if __name__=='__main__':main()
