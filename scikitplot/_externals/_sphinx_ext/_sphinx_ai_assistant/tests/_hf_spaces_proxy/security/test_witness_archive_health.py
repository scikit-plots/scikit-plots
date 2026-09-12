from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

# This fixture intentionally contains compact cryptographic test vectors and
# long, explicit verifier calls; keep flake8 focused on production code.
# flake8: noqa

import base64
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE=Path(__file__).resolve().parent
ROOT=RUNTIME_ROOT
SECURITY=ROOT/"_hf_spaces_proxy"/"security"

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

witness=_load("run162_archive_witness",SECURITY/"witness_archive_health.py")
r161t=_load("run161_helpers_for_run162",HERE/"test_audit_archive_retention.py")
NOW=r161t.NOW

def _canonical(v): return (json.dumps(v,sort_keys=True,separators=(",",":"))+"\n").encode()
def _pub(priv): return base64.b64encode(priv.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)).decode()
def _sig(priv,doc): return base64.b64encode(priv.sign(_canonical(doc))).decode()
def _write(path,doc): path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(_canonical(doc)); return path

def _key(priv,identity,operator,expires=None,channel=None):
    d={"identity":identity,"operator":operator,"expires":witness._ts(expires or (NOW+timedelta(days=900))),"publicKey":_pub(priv)}
    if channel is not None:d["recoveryChannel"]=channel
    return d

def _threshold_root(tmp_path, *, recovery=False, duplicate_operator=False):
    keys={}; privs={}
    prefix="rec" if recovery else "wit"
    for i in range(3):
        kid=f"{prefix}-key-{i+1}"; priv=Ed25519PrivateKey.generate(); privs[kid]=priv
        keys[kid]=_key(priv,f"{prefix}-identity-{i+1}",f"{prefix}-op-a" if duplicate_operator else f"{prefix}-op-{i+1}",NOW+timedelta(days=800),channel=(f"channel-{i+1}" if recovery else None))
    selected=[f"{prefix}-key-1",f"{prefix}-key-2"]
    signed={"_type":"archive-retention-recovery-root" if recovery else "archive-health-witness-root","specVersion":"1.0.0","schemaVersion":1,"rootId":"archive-retention-recovery/main" if recovery else "archive-health-witness/main","version":1,"issuedAt":witness._ts(NOW-timedelta(minutes=1)),"expires":witness._ts(NOW+timedelta(days=700)),"threshold":2,"selectedSignerKeyIds":selected,"keys":{k:keys[k] for k in sorted(keys)}}
    doc={"signed":signed,"signatures":[{"keyId":k,"signature":_sig(privs[k],signed)} for k in selected]}
    path=_write(tmp_path/("recovery-root.json" if recovery else "witness-root.json"),doc)
    return doc,path,witness._sha_bytes(_canonical(doc)),privs

def _run161(tmp_path):
    setup,run160_out,root_doc,root_path,root_pin,root_privs,members,privmap,out,result=r161t._audit(tmp_path)
    return {"setup":setup,"run160":run160_out,"root_doc":root_doc,"root_path":root_path,"root_pin":root_pin,"root_privs":root_privs,"members":members,"privmap":privmap,"out":out,"result":result}

class WitnessAdapter:
    def __init__(self,kid,key,priv,*,now=NOW,bad_view=False,bad_sig=False,reused=False,read_only=True,stale=False): self.kid=kid; self.key=key; self.priv=priv; self.now=now; self.bad_view=bad_view; self.bad_sig=bad_sig; self.reused=reused; self.read_only=read_only; self.stale=stale
    def __call__(self,request):
        view=json.loads(json.dumps(request["expectedView"]))
        if self.bad_view:view["archives"][0]["immutableVersionId"]="split/version"
        observed=self.now-timedelta(hours=2) if self.stale else self.now
        signed={"schemaVersion":1,"operation":"witness-archive-health","sequence":request["sequence"],"witnessIdentity":self.key["identity"],"witnessOperator":self.key["operator"],"challenge":request["challenge"],"selectedWitnessKeyIds":request["selectedWitnessKeyIds"],"run161Artifacts":request["run161Artifacts"],"view":view,"readOnly":self.read_only,"credentialsReused":self.reused,"observedAt":witness._ts(observed)}
        sig=_sig(self.priv,signed)
        if self.bad_sig:sig=base64.b64encode(b"x"*64).decode()
        return {"signed":signed,"signature":{"keyId":self.kid,"signature":sig}}

def _witnessed(tmp_path, *, adapter_mutator=None, root_mutator=None):
    r=_run161(tmp_path/"run161"); wr,wrp,wrpin,wprivs=_threshold_root(tmp_path)
    if root_mutator:
        root_mutator(wr,wprivs); _write(wrp,wr); wrpin=witness._sha_bytes(_canonical(wr))
    adapters=[]
    for kid in ["wit-key-1","wit-key-2","wit-key-3"]:
        adapters.append((kid,WitnessAdapter(kid,wr["signed"]["keys"][kid],wprivs[kid])))
    if adapter_mutator: adapters=adapter_mutator(adapters,wr,wprivs)
    out=tmp_path/"witnessed"
    result=witness.witness_archive_health(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,witnesses=adapters,now=NOW)
    return r,wr,wrp,wrpin,wprivs,out,result

def _new_retention_root(tmp_path, old_root_doc):
    keys={}; privs={}
    for i in range(3):
        kid=f"ret-recovered-{i+1}"; priv=Ed25519PrivateKey.generate(); privs[kid]=priv
        keys[kid]=_key(priv,f"ret-recovered-id-{i+1}",f"ret-recovered-op-{i+1}",NOW+timedelta(days=800))
    selected=["ret-recovered-1","ret-recovered-2"]
    signed={"_type":"archive-retention-root","specVersion":"1.0.0","schemaVersion":1,"rootId":old_root_doc["signed"]["rootId"],"version":1,"issuedAt":witness._ts(NOW),"expires":witness._ts(NOW+timedelta(days=700)),"threshold":2,"selectedSignerKeyIds":selected,"keys":{k:keys[k] for k in sorted(keys)}}
    doc={"signed":signed,"signatures":[{"keyId":k,"signature":_sig(privs[k],signed)} for k in selected]}
    path=_write(tmp_path/"new-retention-root.json",doc)
    return doc,path,witness._sha_bytes(_canonical(doc)),privs

def _recovery_material(tmp_path,r,*,subject_mutator=None,signature_mutator=None,recovery_root_mutator=None,new_root_mutator=None):
    rr,rrp,rrpin,rrprivs=_threshold_root(tmp_path,recovery=True)
    if recovery_root_mutator:
        recovery_root_mutator(rr,rrprivs); _write(rrp,rr); rrpin=witness._sha_bytes(_canonical(rr))
    nr,nrp,nrpin,nrprivs=_new_retention_root(tmp_path,r["root_doc"])
    if new_root_mutator:
        new_root_mutator(nr,nrprivs); _write(nrp,nr); nrpin=witness._sha_bytes(_canonical(nr))
    state=json.loads((r["out"]/"trusted-archive-health-state.json").read_text())
    selected=["rec-key-1","rec-key-2"]
    subject={"_type":"archive-retention-root-recovery","specVersion":"1.0.0","schemaVersion":1,"recoveryId":"retention-recovery/1","issuedAt":witness._ts(NOW),"recoveryRootSha256":rrpin,"oldRetentionRootSha256":r["root_pin"],"newRetentionRootSha256":nrpin,"run161HealthChainHeadSha256":state["healthChainHeadSha256"],"activeArchiveIds":state["activeArchiveIds"],"compromisedOldKeyIds":["ret-root-1"],"selectedRecoveryKeyIds":selected,"retirementAuthorizedArchiveIds":[]}
    if subject_mutator: subject_mutator(subject)
    sp=_write(tmp_path/"recovery-subject.json",subject)
    sigs={"schemaVersion":1,"signatures":[{"keyId":k,"channel":rr["signed"]["keys"][k]["recoveryChannel"],"signature":_sig(rrprivs[k],subject)} for k in selected]}
    if signature_mutator: signature_mutator(sigs,rr,rrprivs,subject)
    sigp=_write(tmp_path/"recovery-signatures.json",sigs)
    return rr,rrp,rrpin,nr,nrp,nrpin,sp,sigp

def _recover(tmp_path,**kwargs):
    r=_run161(tmp_path/"run161")
    rr,rrp,rrpin,nr,nrp,nrpin,sp,sigp=_recovery_material(tmp_path,r,**kwargs)
    out=tmp_path/"recovered"
    result=witness.recover_retention_root(run160_dir=r["run160"],run161_dir=r["out"],old_retention_root_path=r["root_path"],old_retention_root_pin=r["root_pin"],new_retention_root_path=nrp,recovery_root_path=rrp,recovery_root_pin=rrpin,recovery_subject_path=sp,recovery_signatures_path=sigp,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW)
    return r,rr,nr,out,result

def test_run162_witness_bootstrap_and_offline_verify(tmp_path):
    r,wr,wrp,wrpin,_,out,result=_witnessed(tmp_path)
    assert result["ok"] and result["witness_count"]==3
    v=witness.verify_witness_history(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW)
    assert v["witness_chain_head_sha256"]==result["witness_chain_head_sha256"]

def test_run162_two_of_three_witness_quorum_is_allowed(tmp_path):
    def mt(a,wr,p):return a[:2]
    *_,result=_witnessed(tmp_path,adapter_mutator=mt); assert result["witness_count"]==2

def test_run162_any_observed_split_view_fails_closed(tmp_path):
    def mt(a,wr,p):
        a[2]=(a[2][0],WitnessAdapter(a[2][0],wr["signed"]["keys"][a[2][0]],p[a[2][0]],bad_view=True)); return a
    with pytest.raises(witness.ArchiveWitnessError,match="SPLIT_VIEW"):_witnessed(tmp_path,adapter_mutator=mt)

def test_run162_witness_signature_mutation_fails(tmp_path):
    def mt(a,wr,p):a[0]=(a[0][0],WitnessAdapter(a[0][0],wr["signed"]["keys"][a[0][0]],p[a[0][0]],bad_sig=True));return a
    with pytest.raises(witness.ArchiveWitnessError,match="SIGNATURE_INVALID"):_witnessed(tmp_path,adapter_mutator=mt)

def test_run162_witness_must_be_read_only_and_not_reuse_credentials(tmp_path):
    def mt(a,wr,p):a[0]=(a[0][0],WitnessAdapter(a[0][0],wr["signed"]["keys"][a[0][0]],p[a[0][0]],reused=True));return a
    with pytest.raises(witness.ArchiveWitnessError,match="AUTHORITY_INVALID"):_witnessed(tmp_path,adapter_mutator=mt)

def test_run162_stale_witness_observation_fails(tmp_path):
    def mt(a,wr,p):a[0]=(a[0][0],WitnessAdapter(a[0][0],wr["signed"]["keys"][a[0][0]],p[a[0][0]],stale=True));return a
    with pytest.raises(witness.ArchiveWitnessError,match="NOT_FRESH|BEFORE_ROOT"):_witnessed(tmp_path,adapter_mutator=mt)

def test_run162_witness_root_pin_is_mandatory(tmp_path):
    r,wr,wrp,wrpin,_,out,_=_witnessed(tmp_path)
    with pytest.raises(witness.ArchiveWitnessError,match="PIN_MISMATCH"):
        witness.verify_witness_history(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin="0"*64,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW)

def test_run162_witness_root_extra_signature_is_rejected(tmp_path):
    def rm(doc,privs):doc["signatures"].append({"keyId":"wit-key-3","signature":_sig(privs["wit-key-3"],doc["signed"])})
    with pytest.raises(witness.ArchiveWitnessError,match="SIGNATURE_SET_INVALID"):_witnessed(tmp_path,root_mutator=rm)

def test_run162_witness_chain_mutation_is_detected(tmp_path):
    r,wr,wrp,wrpin,_,out,_=_witnessed(tmp_path)
    d=json.loads((out/"release-archive-witness-bundle.json").read_text());d["events"][0]["view"]["archives"][0]["retentionMode"]="mutated";_write(out/"release-archive-witness-bundle.json",d)
    with pytest.raises(witness.ArchiveWitnessError):witness.verify_witness_history(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW,historical=True)

def test_run162_output_is_create_only(tmp_path):
    r,wr,wrp,wrpin,wprivs,out,_=_witnessed(tmp_path)
    adapters=[(k,WitnessAdapter(k,wr["signed"]["keys"][k],wprivs[k])) for k in ["wit-key-1","wit-key-2"]]
    with pytest.raises(witness.ArchiveWitnessError,match="OUTPUT_EXISTS"):
        witness.witness_archive_health(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,witnesses=adapters,now=NOW)

def test_run162_recovery_succeeds_without_old_root_signatures(tmp_path):
    r,rr,nr,out,result=_recover(tmp_path)
    assert result["ok"] and result["retirement_authorized"]==[]
    assert (out/"recovered-retention-root.json").read_bytes()==_canonical(nr)

def test_run162_recovery_explicitly_forbids_retirement(tmp_path):
    with pytest.raises(witness.ArchiveWitnessError,match="RETIREMENT_FORBIDDEN"):_recover(tmp_path,subject_mutator=lambda s:s.__setitem__("retirementAuthorizedArchiveIds",["archive-x"]))

def test_run162_recovery_requires_compromised_old_key(tmp_path):
    with pytest.raises(witness.ArchiveWitnessError,match="COMPROMISED_KEYS_INVALID"):_recover(tmp_path,subject_mutator=lambda s:s.__setitem__("compromisedOldKeyIds",[]))

def test_run162_recovery_signature_mutation_fails(tmp_path):
    def sm(sigs,rr,privs,subject):sigs["signatures"][0]["signature"]=base64.b64encode(b"z"*64).decode()
    with pytest.raises(witness.ArchiveWitnessError,match="SIGNATURE_INVALID"):_recover(tmp_path,signature_mutator=sm)

def test_run162_recovery_channel_binding_is_cryptographic(tmp_path):
    def sm(sigs,rr,privs,subject):sigs["signatures"][0]["channel"]="channel-3"
    with pytest.raises(witness.ArchiveWitnessError,match="SIGNATURE_BINDING_INVALID"):_recover(tmp_path,signature_mutator=sm)

def test_run162_recovery_root_extra_signature_is_rejected(tmp_path):
    def rm(doc,privs):doc["signatures"].append({"keyId":"rec-key-3","signature":_sig(privs["rec-key-3"],doc["signed"])})
    with pytest.raises(witness.ArchiveWitnessError,match="SIGNATURE_SET_INVALID"):_recover(tmp_path,recovery_root_mutator=rm)

def test_run162_recovery_health_chain_binding_is_exact(tmp_path):
    with pytest.raises(witness.ArchiveWitnessError,match="HEALTH_BINDING_INVALID"):_recover(tmp_path,subject_mutator=lambda s:s.__setitem__("run161HealthChainHeadSha256","0"*64))

def test_run162_recovery_new_root_hash_binding_is_exact(tmp_path):
    with pytest.raises(witness.ArchiveWitnessError,match="ROOT_BINDING_INVALID"):_recover(tmp_path,subject_mutator=lambda s:s.__setitem__("newRetentionRootSha256","0"*64))

def test_run162_recovery_authority_must_be_separate(tmp_path):
    def rm(doc,privs):
        # make one selected recovery operator collide with the old retention authority
        doc["signed"]["keys"]["rec-key-1"]["operator"]="ret-root-op-1";doc["signatures"]=[{"keyId":k,"signature":_sig(privs[k],doc["signed"])} for k in doc["signed"]["selectedSignerKeyIds"]]
    with pytest.raises(witness.ArchiveWitnessError,match="AUTHORITY_SEPARATION_INVALID"):_recover(tmp_path,recovery_root_mutator=rm)

def test_run162_recovery_is_deterministic(tmp_path):
    r=_run161(tmp_path/"run161");rr,rrp,rrpin,nr,nrp,nrpin,sp,sigp=_recovery_material(tmp_path,r)
    outs=[]
    for name in ("a","b"):
        out=tmp_path/name;witness.recover_retention_root(run160_dir=r["run160"],run161_dir=r["out"],old_retention_root_path=r["root_path"],old_retention_root_pin=r["root_pin"],new_retention_root_path=nrp,recovery_root_path=rrp,recovery_root_pin=rrpin,recovery_subject_path=sp,recovery_signatures_path=sigp,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW);outs.append(out)
    assert all((outs[0]/n).read_bytes()==(outs[1]/n).read_bytes() for n in witness._RECOVERY_OUTPUT_NAMES)

def test_run162_recovery_output_has_no_private_keys_or_retirement_authority(tmp_path):
    *_,out,result=_recover(tmp_path)
    text="\n".join(p.read_text() for p in out.iterdir()).lower()
    assert "privatekey" not in text and "private key" not in text and '"retirementauthorizedarchiveids":[]' in text.replace(" ","")

def test_run162_duplicate_json_keys_are_rejected(tmp_path):
    p=tmp_path/"dup.json";p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(witness.ArchiveWitnessError,match="DUPLICATE_KEY"):witness._read_json(p,"DUP")

def test_run162_command_adapter_bounds_output_while_produced(tmp_path, monkeypatch):
    script=tmp_path/"noisy.py";script.write_text("import sys\nsys.stdout.write('x'*(9*1024*1024))\n")
    real_popen = witness.subprocess.Popen
    seen = {}

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(witness.subprocess, "Popen", tracking_popen)
    adapter=witness.command_witness([sys.executable,str(script)])
    with pytest.raises(witness.ArchiveWitnessError,match="OUTPUT_TOO_LARGE"):adapter({"x":1})
    proc = seen["proc"]
    assert proc.poll() is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed

def test_run162_documentation_describes_split_view_witnessing_and_recovery():
    guide=(SECURITY/"RELEASE_ARCHIVE_WITNESS_GUIDE.md").read_text();gates=(SECURITY/"SECURITY_RELEASE_GATES.md").read_text()
    for phrase in ("split view","witness quorum","out-of-band","retention root recovery","retirement"):
        assert phrase in guide.lower()
    assert "Run 162" in gates


def test_run162_witness_authority_is_separate_from_archive_planes(tmp_path):
    def rm(doc,privs):
        doc["signed"]["keys"]["wit-key-1"]["operator"]="ret-root-op-1"
        doc["signatures"]=[{"keyId":k,"signature":_sig(privs[k],doc["signed"])} for k in doc["signed"]["selectedSignerKeyIds"]]
    with pytest.raises(witness.ArchiveWitnessError,match="PLANES_OVERLAP"):_witnessed(tmp_path,root_mutator=rm)

def test_run162_witness_history_advances_across_new_run161_audit(tmp_path):
    r,wr,wrp,wrpin,wprivs,wout,_=_witnessed(tmp_path/"first")
    state=json.loads((r["out"]/"trusted-archive-health-state.json").read_text()); receipt=json.loads((r["run160"]/"release-native-evidence-archive-receipt.json").read_text()); later=NOW+timedelta(minutes=5)
    mp2,_=r161t._membership(tmp_path,r["root_doc"],r["root_privs"],r["run160"],receipt,r["members"],sequence=2,previous_head=state["healthChainHeadSha256"],previous_members=r["members"],issued=later,name="membership-2.json")
    targets=[]
    for m in r["members"]:
        pp,ap=r["privmap"][m["archiveId"]]; p=r161t.Provider(m,pp,now=later); targets.append((m["archiveId"],p,r161t.Auditor(m,ap,p,now=later)))
    run161b=tmp_path/"run161-second"
    r161t.health.audit_archive_health(run160_dir=r["run160"],output_dir=run161b,retention_root_path=r["root_path"],membership_path=mp2,targets=targets,expected_retention_root_sha256=r["root_pin"],expected_bootstrap_root_sha256=r["setup"]["pin"],expected_recovery_root_sha256=r["setup"]["rr_pin"],expected_attestation_root_sha256=[r["setup"]["ca_pin"]],previous_output_dir=r["out"],now=later)
    adapters=[(k,WitnessAdapter(k,wr["signed"]["keys"][k],wprivs[k],now=later)) for k in ["wit-key-1","wit-key-2","wit-key-3"]]
    out2=tmp_path/"witness-second"
    res=witness.witness_archive_health(run160_dir=r["run160"],run161_dir=run161b,retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out2,witnesses=adapters,previous_output_dir=wout,now=later)
    assert res["sequence"]==2
    bundle=json.loads((out2/"release-archive-witness-bundle.json").read_text()); assert bundle["events"][0]["run161HealthChainHeadSha256"]!=bundle["events"][1]["run161HealthChainHeadSha256"]

def test_run162_recovery_output_offline_verifier_replays(tmp_path):
    r,rr,nr,out,result=_recover(tmp_path)
    rr_path=tmp_path/"recovery-root.json"; rr_pin=witness._sha_bytes(_canonical(rr))
    v=witness.verify_retention_root_recovery(run160_dir=r["run160"],run161_dir=r["out"],old_retention_root_path=r["root_path"],old_retention_root_pin=r["root_pin"],recovery_root_path=rr_path,recovery_root_pin=rr_pin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW)
    assert v["ok"] and v["retirement_authorized"]==[]

def test_run162_recovery_output_mutation_is_detected(tmp_path):
    r,rr,nr,out,result=_recover(tmp_path)
    rec=json.loads((out/"retention-root-recovery-receipt.json").read_text());rec["retirementAuthorizedArchiveIds"]=["x"];_write(out/"retention-root-recovery-receipt.json",rec)
    with pytest.raises(witness.ArchiveWitnessError,match="RECEIPT_MISMATCH"):
        witness.verify_retention_root_recovery(run160_dir=r["run160"],run161_dir=r["out"],old_retention_root_path=r["root_path"],old_retention_root_pin=r["root_pin"],recovery_root_path=tmp_path/"recovery-root.json",recovery_root_pin=witness._sha_bytes(_canonical(rr)),bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW)

def test_run162_recovery_rejects_old_operator_reuse_in_new_root(tmp_path):
    def nm(doc,privs):
        doc["signed"]["keys"]["ret-recovered-1"]["operator"]="ret-root-op-1"
        doc["signatures"]=[{"keyId":k,"signature":_sig(privs[k],doc["signed"])} for k in doc["signed"]["selectedSignerKeyIds"]]
    with pytest.raises(witness.ArchiveWitnessError,match="AUTHORITY_SEPARATION_INVALID"):_recover(tmp_path,new_root_mutator=nm)

def test_run162_recovery_subject_must_be_inside_recovery_root_lifetime(tmp_path):
    with pytest.raises(witness.ArchiveWitnessError,match="OUTSIDE_RECOVERY_ROOT_LIFETIME"):_recover(tmp_path,subject_mutator=lambda s:s.__setitem__("issuedAt",witness._ts(NOW-timedelta(days=10))))


def test_run162_witness_set_cannot_be_rewritten_by_dropping_response(tmp_path):
    r,wr,wrp,wrpin,_,out,_=_witnessed(tmp_path)
    receipt=json.loads((out/"release-archive-witness-receipt.json").read_text());receipt["events"][0]["responses"]=receipt["events"][0]["responses"][:2];_write(out/"release-archive-witness-receipt.json",receipt)
    bundle=json.loads((out/"release-archive-witness-bundle.json").read_text());bundle["events"][0]["witnessKeyIds"]=bundle["events"][0]["witnessKeyIds"][:2];bundle["events"][0]["witnessResponseSha256s"]=bundle["events"][0]["witnessResponseSha256s"][:2];_write(out/"release-archive-witness-bundle.json",bundle)
    with pytest.raises(witness.ArchiveWitnessError):
        witness.verify_witness_history(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW,historical=True)

def test_run162_witness_state_rejects_extra_fields(tmp_path):
    r,wr,wrp,wrpin,_,out,_=_witnessed(tmp_path)
    state=json.loads((out/"trusted-archive-witness-state.json").read_text());state["unexpected"]=True;_write(out/"trusted-archive-witness-state.json",state)
    with pytest.raises(witness.ArchiveWitnessError,match="STATE_BINDING_INVALID"):
        witness.verify_witness_history(run160_dir=r["run160"],run161_dir=r["out"],retention_root_path=r["root_path"],retention_root_pin=r["root_pin"],witness_root_path=wrp,witness_root_pin=wrpin,bootstrap_pin=r["setup"]["pin"],recovery_pin=r["setup"]["rr_pin"],attestation_pins=[r["setup"]["ca_pin"]],output_dir=out,now=NOW,historical=True)
