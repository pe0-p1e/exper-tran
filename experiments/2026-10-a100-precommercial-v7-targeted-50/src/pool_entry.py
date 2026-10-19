"""Concurrent child entry: unchanged algorithms, serialized supplementary audits."""
import argparse
import fcntl
import sys
from supplement_registry import install,memory_cap,verify_existing_data
install();memory_cap()
import vision_features
import strict_model_audit
from v7_common import OUT
original_audit=vision_features.audit_loaded_model


def audit_serialized(*args,**kwargs):
    path=OUT/'multiproxy_8h_supplement/audits/model_audit_writer.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return original_audit(*args,**kwargs)


vision_features.audit_loaded_model=audit_serialized
strict_model_audit.audit_loaded_model=audit_serialized
mode=sys.argv.pop(1)
if mode=='attack':
    import run_v7_attack
    run_v7_attack.prepare_data=verify_existing_data
    run_v7_attack.main()
elif mode=='calibration':
    import supplement_registry
    supplement_registry.install=lambda:__import__('v7_common').REGISTRY
    import smoke_supplement_models
    smoke_supplement_models.main()
elif mode=='evaluation':
    import evaluate_v7
    evaluate_v7.main()
else:raise ValueError('Unknown pool phase')
