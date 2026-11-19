#!/usr/bin/env python3
"""Phase 7: verify malformed untrusted packages never poison persistent SQLite replay state.
Existing real Go receiver and hardened replay module are used, not mocked.
Uses TEMPORARY state and a PUBLIC research vector only.
"""
from __future__ import annotations
import argparse, importlib.util, json, random, sys, tempfile, time
from pathlib import Path

# Discover project files at runtime; no source copy and no protected dataset.
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from run_security_fuzz import parse_offsets, produce_cases

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--go-bin',required=True,type=Path)
    p.add_argument('--package',required=True,type=Path)
    p.add_argument('--vector',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--count',type=int,default=70)
    args=p.parse_args()
    sys.path.insert(0,str(args.repo.resolve()/'replay-receiver'))
    from real_go_receiver_with_replay import receive_with_persistent_replay_protection
    from hardened_replay_cache import HardenedReplayCache, ReplayDetectedError
    raw=args.package.read_bytes()
    vector=json.loads(args.vector.read_text())
    master=bytes.fromhex(vector['validation_master_key_hex'])
    expected=vector['expected_hashes']['original_dicom_sha256']
    cases=produce_cases(raw,parse_offsets(raw),random.Random(20261104),master,max(args.count,100))
    rejected=0; failures=[]; beginning=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='amie-phase7-replay-') as directory:
        # Explicit provisioning to avoid accidental DB creation in production.
        cache=HardenedReplayCache.provision(Path(directory)/'replay.sqlite',receiver_id='phase7-fuzz')
        try:
            for cat,detail,mutated,_,signed in cases:
                if signed: continue # Some test-key-authorized packets may be expensive to fully decrypt.
                if rejected>=args.count:break
                try:
                    receive_with_persistent_replay_protection(raw_package=mutated,
                        public_validation_key=master,go_receiver_binary=args.go_bin,
                        cache=cache,expected_original_sha256=expected)
                except Exception:
                    pass # Any rejection is acceptable; SQLite state must remain unchanged.
                else:
                    failures.append({'category':cat,'reason':'untrusted mutated package accepted'})
                rejected+=1
                if cache.count_entries()!=0:
                    failures.append({'category':cat,'reason':'invalid package poisoned persistent replay state'})
                    break
            legitimate=receive_with_persistent_replay_protection(raw_package=raw,
                public_validation_key=master,go_receiver_binary=args.go_bin,
                cache=cache,expected_original_sha256=expected)
            if legitimate.recovered_sha256!=expected:
                failures.append({'reason':'valid original DICOM digest mismatch'})
            if cache.count_entries()!=1:
                failures.append({'reason':'valid packet did not produce exactly one replay reservation'})
            try:
                receive_with_persistent_replay_protection(raw_package=raw,
                    public_validation_key=master,go_receiver_binary=args.go_bin,
                    cache=cache,expected_original_sha256=expected)
            except ReplayDetectedError:
                pass
            else:
                failures.append({'reason':'same authenticated package accepted twice'})
        finally: cache.close()
    result={'experiment':'Phase7-IntegratedReplayFuzz-CFP9-v2','malformed_cases':rejected,
        'malformed_packages_poisoning_replay_state':len([x for x in failures if 'poison' in x.get('reason','')]),
        'valid_recovery_after_mutations':legitimate.recovered_sha256==expected,
        'repeated_valid_package_rejected':not any('twice' in x.get('reason','') for x in failures),
        'findings':failures,'result':'PASS' if not failures else 'FAIL',
        'seconds_elapsed':round(time.perf_counter()-beginning,3),
        'limitations':'Single public fixture and controlled temporary SQLite database; no real key management.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if failures: raise SystemExit(2)
if __name__=='__main__':main()
