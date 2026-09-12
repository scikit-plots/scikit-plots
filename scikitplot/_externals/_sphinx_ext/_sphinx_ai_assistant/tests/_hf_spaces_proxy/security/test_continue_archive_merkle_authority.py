from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
import copy
import json
import sys
from datetime import timedelta
from pathlib import Path
import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

TESTS = Path(__file__).resolve().parent
SEC = RUNTIME_ROOT / '_hf_spaces_proxy' / 'security'
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SEC))
import test_verify_archive_merkle_transparency as t164  # type: ignore[import-not-found]
import test_govern_archive_merkle_log_authority as t165  # type: ignore[import-not-found]
import continue_archive_merkle_authority as cont  # type: ignore[import-not-found]
import govern_archive_merkle_log_authority as auth  # type: ignore[import-not-found]
import verify_archive_merkle_transparency as merkle  # type: ignore[import-not-found]

NOW = t164.NOW; LATER = NOW + timedelta(minutes=5)

def _pub(p):
    return base64.b64encode(
        p.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()


def _write(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(cont._canonical(o))
    return p

def _advance_run163(
    f, tmp, *, seq, run161_prev, wout_prev, aout_prev, now
):
    tmp.mkdir(parents=True, exist_ok=True)
    r = f['r']
    state = json.loads(
        (run161_prev / 'trusted-archive-health-state.json').read_text()
    )
    receipt = json.loads(
        (r['run160'] / 'release-native-evidence-archive-receipt.json').read_text()
    )
    mp, _ = t164.t163.r162t.r161t._membership(
        tmp,
        r['root_doc'],
        r['root_privs'],
        r['run160'],
        receipt,
        r['members'],
        sequence=seq,
        previous_head=state['healthChainHeadSha256'],
        previous_members=r['members'],
        issued=now,
        name=f'membership-{seq}.json',
    )
    targets = []
    for m in r['members']:
        ppv, ap = r['privmap'][m['archiveId']]
        provider = t164.t163.r162t.r161t.Provider(m, ppv, now=now)
        targets.append(
            (
                m['archiveId'],
                provider,
                t164.t163.r162t.r161t.Auditor(m, ap, provider, now=now),
            )
        )
    run161 = tmp / f'run161-{seq}'
    t164.t163.r162t.r161t.health.audit_archive_health(
        run160_dir=r['run160'],
        output_dir=run161,
        retention_root_path=r['root_path'],
        membership_path=mp,
        targets=targets,
        expected_retention_root_sha256=r['root_pin'],
        expected_bootstrap_root_sha256=r['setup']['pin'],
        expected_recovery_root_sha256=r['setup']['rr_pin'],
        expected_attestation_root_sha256=[r['setup']['ca_pin']],
        previous_output_dir=run161_prev,
        now=now,
    )
    wad = [
        (
            k,
            t164.t163.r162t.WitnessAdapter(
                k, f['wr']['signed']['keys'][k], f['wpriv'][k], now=now
            ),
        )
        for k in ['wit-key-1', 'wit-key-2', 'wit-key-3']
    ]
    wout = tmp / f'witness-{seq}'
    t164.t163.r162t.witness.witness_archive_health(
        run160_dir=r['run160'],
        run161_dir=run161,
        retention_root_path=r['root_path'],
        retention_root_pin=r['root_pin'],
        witness_root_path=f['wrp'],
        witness_root_pin=f['wrpin'],
        bootstrap_pin=r['setup']['pin'],
        recovery_pin=r['setup']['rr_pin'],
        attestation_pins=[r['setup']['ca_pin']],
        output_dir=wout,
        witnesses=wad,
        previous_output_dir=wout_prev,
        now=now,
    )
    cad = [
        (
            cid,
            t164.t163.Channel(cid, cfg, f['cpriv'][cid], now),
            t164.t163.Observer(cid, cfg, f['opriv'][cid], now),
        )
        for cid, cfg in f['plan']['signed']['channels'].items()
    ]
    aout = tmp / f'anchor-{seq}'
    t164.t163.anchor.anchor_archive_health(
        run160_dir=r['run160'],
        run161_dir=run161,
        retention_root_path=r['root_path'],
        retention_root_pin=r['root_pin'],
        witness_root_path=f['wrp'],
        witness_root_pin=f['wrpin'],
        bootstrap_pin=r['setup']['pin'],
        recovery_pin=r['setup']['rr_pin'],
        attestation_pins=[r['setup']['ca_pin']],
        run162_dir=wout,
        anchor_plan_path=f['pp'],
        output_dir=aout,
        channels=cad,
        previous_output_dir=aout_prev,
        now=now,
    )
    return run161, wout, aout

@pytest.fixture(scope='module')
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('run166-base')
    f = t164._fixture(tmp / 'pred')
    run164 = tmp / 'run164'
    merkle.anchor_merkle_transparency(
        **t164._kwargs(f, run164), adapters=f['adapters']
    )
    gov, gp, gpin, gpriv = t165._control_root(tmp, name='gov')
    rec, rp, rpin, rpriv = t165._control_root(
        tmp, recovery=True, name='recovery'
    )
    b = {
        'tmp': tmp,
        'f': f,
        'run164': run164,
        'gov': gov,
        'gp': gp,
        'gpin': gpin,
        'gpriv': gpriv,
        'rec': rec,
        'rp': rp,
        'rpin': rpin,
        'rpriv': rpriv,
    }
    run165, _, _ = t165._bootstrap(b, tmp / 'bootstrap')
    r = f['r']
    run161b, wout2, aout2 = _advance_run163(
        f,
        tmp / 'advance2',
        seq=2,
        run161_prev=r['out'],
        wout_prev=f['wout'],
        aout_prev=f['aout'],
        now=LATER,
    )
    b.update(run165=run165, run161b=run161b, wout2=wout2, aout2=aout2)
    return b

def _adapters(base, authority=None, log_privs=None, gossip_privs=None, now=LATER):
    f = base['f']
    authority = authority or json.loads(
        (base['run165'] / 'active-archive-log-authority.json').read_text()
    )['authority']
    log_privs = log_privs or f['lpriv']
    gossip_privs = gossip_privs or f['gpriv']
    out = []
    for lid, cfg in authority.items():
        la = t164.LogAdapter(lid, cfg, log_privs[lid], now)
        la.leaves = list(f['logs'][lid].leaves)
        ga = t164.GossipAdapter(lid, cfg, gossip_privs[lid], now)
        out.append((lid, la, ga))
    return out

def _kwargs(
    base,
    out,
    *,
    run161=None,
    wout=None,
    aout=None,
    run165=None,
    now=LATER,
    historical=None,
):
    f = base['f']
    r = f['r']
    d = dict(
        run160_dir=r['run160'],
        run161_dir=run161 or base['run161b'],
        retention_root_path=r['root_path'],
        retention_root_pin=r['root_pin'],
        witness_root_path=f['wrp'],
        witness_root_pin=f['wrpin'],
        bootstrap_pin=r['setup']['pin'],
        recovery_pin=r['setup']['rr_pin'],
        attestation_pins=[r['setup']['ca_pin']],
        run162_dir=wout or base['wout2'],
        anchor_plan_path=f['pp'],
        run163_dir=aout or base['aout2'],
        run164_dir=base['run164'],
        transparency_root_path=f['rp'],
        transparency_root_pin=f['rpin'],
        governance_root_path=base['gp'],
        governance_root_pin=base['gpin'],
        recovery_root_path=base['rp'],
        recovery_root_pin=base['rpin'],
        run165_dir=run165 or base['run165'],
        output_dir=out,
        now=now,
    )
    if historical is not None:
        d['historical'] = historical
    return d

def _append(base, tmp, **kw):
    out = tmp / 'out'
    args = _kwargs(
        base,
        out,
        **{
            k: v
            for k, v in kw.items()
            if k in {'run161', 'wout', 'aout', 'run165', 'now'}
        },
    )
    args.pop('historical', None)
    r165 = kw.get('run165') or base['run165']
    authority_map = json.loads(
        (r165 / 'active-archive-log-authority.json').read_text()
    )['authority']
    adapters = kw.get('adapters') or _adapters(base, authority=authority_map)
    res = cont.continue_merkle_authority(
        **args,
        adapters=adapters,
        previous_output_dir=kw.get('previous'),
    )
    return out, res

def test_run166_bootstrap_bridge_and_offline_verify(base, tmp_path):
    out, res = _append(base, tmp_path)
    assert res['sequence'] == 2 and res['post_handoff_sequence'] == 1
    assert cont.verify_merkle_authority_continuity(**_kwargs(base, out))['ok']

def test_run166_first_checkpoint_bridges_exact_run165_checkpoint(base, tmp_path):
    out, _ = _append(base, tmp_path)
    rec = json.loads(
        (out / 'release-archive-merkle-continuity-receipt.json').read_text()
    )
    st = json.loads(
        (base['run165'] / 'trusted-archive-log-authority-state.json').read_text()
    )
    for lid, doc in rec['events'][0]['logResponses'].items():
        assert (
            doc['signed']['previousTreeSize']
            == st['continuityCheckpoints'][lid]['treeSize']
        )
        assert (
            doc['signed']['previousRootHash']
            == st['continuityCheckpoints'][lid]['rootHash']
        )
        assert doc['signed']['consistencyProof']

def _rotated_run165(base, tmp, kind='scheduled-rotation'):
    stp = base['run165'] / 'trusted-archive-log-authority-state.json'
    st = json.loads(stp.read_text())
    cur = copy.deepcopy(st['activeAuthority'])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[0]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]['publicKey'] = _pub(lp)
    nxt[lid]['gossipPublicKey'] = _pub(gp)
    compromised = (
        [auth._pub_fingerprint(cur[lid]['publicKey'])]
        if kind == 'compromise-recovery'
        else None
    )
    tr, info = t165._transition(
        base,
        tmp,
        sequence=2,
        kind=kind,
        current=cur,
        nxt=nxt,
        previous_state_path=stp,
        compromised=compromised,
        issued=NOW + timedelta(seconds=2),
    )
    logs = dict(base['f']['lpriv'])
    goss = dict(base['f']['gpriv'])
    logs[lid] = lp
    goss[lid] = gp
    hf = t165._handoff(
        base,
        tr,
        info,
        new_privs={'log': logs, 'gossip': goss},
        old_privs=(
            {'log': base['f']['lpriv'], 'gossip': base['f']['gpriv']}
            if kind == 'scheduled-rotation'
            else None
        ),
    )
    tp = _write(tmp / 'tr.json', tr)
    hp = _write(tmp / 'hf.json', hf)
    out = tmp / 'run165-2'
    auth.apply_log_authority_transition(
        run164_dir=base['run164'],
        transparency_root_path=base['f']['rp'],
        transparency_root_pin=base['f']['rpin'],
        governance_root_path=base['gp'],
        governance_root_pin=base['gpin'],
        recovery_root_path=base['rp'],
        recovery_root_pin=base['rpin'],
        transition_path=tp,
        handoff_path=hp,
        previous_output_dir=base['run165'],
        output_dir=out,
        now=NOW + timedelta(seconds=2),
    )
    return out, lid, logs, goss

def test_run166_scheduled_rotation_appends_with_new_key_only(base, tmp_path):
    r165, _, logs, goss = _rotated_run165(base, tmp_path / 'r')
    authority = json.loads(
        (r165 / 'active-archive-log-authority.json').read_text()
    )['authority']
    adapters = _adapters(base, authority, logs, goss)
    _, res = _append(
        base,
        tmp_path / 'a',
        run165=r165,
        adapters=adapters,
    )
    assert res['sequence'] == 2

    bad = _adapters(base, authority, base['f']['lpriv'], base['f']['gpriv'])
    with pytest.raises(cont.ArchiveMerkleContinuityError):
        cont.continue_merkle_authority(
            **_kwargs(base, tmp_path / 'bad', run165=r165),
            adapters=bad,
        )

def test_run166_compromise_recovery_never_requires_old_key(base, tmp_path):
    r165, _, logs, goss = _rotated_run165(
        base, tmp_path / 'r', kind='compromise-recovery'
    )
    authority = json.loads(
        (r165 / 'active-archive-log-authority.json').read_text()
    )['authority']
    _, res = _append(
        base,
        tmp_path / 'a',
        run165=r165,
        adapters=_adapters(base, authority, logs, goss),
    )
    assert res['sequence'] == 2

def test_run166_same_run163_epoch_cannot_repeat(base, tmp_path):
    out, _ = _append(base, tmp_path / 'a')
    with pytest.raises(
        cont.ArchiveMerkleContinuityError, match='SEQUENCE_INVALID'
    ):
        cont.continue_merkle_authority(
            **_kwargs(base, tmp_path / 'b'),
            adapters=_adapters(base),
            previous_output_dir=out,
        )

def test_run166_run165_mutation_detected(base, tmp_path):
    bad = tmp_path / 'run165'
    import shutil

    shutil.copytree(base['run165'], bad)
    p = bad / 'active-archive-log-authority.json'
    d = json.loads(p.read_text())
    d['sequence'] = 99
    p.write_bytes(cont._canonical(d))
    with pytest.raises(cont.ArchiveMerkleContinuityError):
        cont.continue_merkle_authority(
            **_kwargs(base, tmp_path / 'out', run165=bad),
            adapters=_adapters(base),
        )

def test_run166_log_signature_mutation_detected(base, tmp_path):
    out, _ = _append(base, tmp_path / 'a')
    p = out / 'release-archive-merkle-continuity-receipt.json'
    d = json.loads(p.read_text())
    d['events'][0]['logResponses']['merkle-log-1']['signature'] = 'A' * 88
    p.write_bytes(cont._canonical(d))
    with pytest.raises(cont.ArchiveMerkleContinuityError):
        cont.verify_merkle_authority_continuity(**_kwargs(base, out))

def test_run166_gossip_split_view_rejected(base, tmp_path):
    ads = _adapters(base)
    lid = 'merkle-log-3'
    cfg = json.loads(
        (base['run165'] / 'active-archive-log-authority.json').read_text()
    )['authority'][lid]

    def mut(s):
        s['checkpointSha256s'][next(iter(s['checkpointSha256s']))] = '0' * 64

    ads = [
        (
            x,
            la,
            (
                t164.GossipAdapter(x, cfg, base['f']['gpriv'][x], LATER, mut)
                if x == lid
                else ga
            ),
        )
        for x, la, ga in ads
    ]
    with pytest.raises(
        cont.ArchiveMerkleContinuityError, match='SPLIT_VIEW'
    ):
        cont.continue_merkle_authority(
            **_kwargs(base, tmp_path / 'out'), adapters=ads
        )

def test_run166_duplicate_json_keys_rejected(tmp_path):
    p = tmp_path / 'x.json'
    p.write_text('{"a": 1, "a": 2}\n')
    with pytest.raises(
        cont.ArchiveMerkleContinuityError, match='DUPLICATE_KEY'
    ):
        cont._read_json(p, 'X')

def test_run166_output_exists_rejected(base, tmp_path):
    out, _ = _append(base, tmp_path / 'a')
    with pytest.raises(
        cont.ArchiveMerkleContinuityError, match='OUTPUT_EXISTS'
    ):
        cont.continue_merkle_authority(
            **_kwargs(base, out), adapters=_adapters(base)
        )

def test_run166_live_active_epoch_expires_but_historical_survives(base, tmp_path):
    out, _ = _append(base, tmp_path / 'a')
    with pytest.raises(
        cont.ArchiveMerkleContinuityError, match='ACTIVE_EPOCH_STALE'
    ):
        cont.verify_merkle_authority_continuity(
            **_kwargs(base, out, now=LATER + timedelta(days=40))
        )
    assert cont.verify_merkle_authority_continuity(
        **_kwargs(base, out, now=LATER + timedelta(days=40), historical=True)
    )['ok']

def test_run166_documentation_mentions_recovered_authority_rfc6962_and_retired_key():
    text = (SEC / 'RELEASE_ARCHIVE_MERKLE_CONTINUITY_GUIDE.md').read_text().lower()
    for phrase in (
        'run 165',
        'rfc6962',
        'retired key',
        'consistency proof',
        'compromise recovery',
    ):
        assert phrase in text

def test_run166_second_post_handoff_epoch_extends_same_trees(base, tmp_path):
    ads = _adapters(base); out1 = tmp_path / 'out1'; args = _kwargs(base, out1); args.pop('historical', None)
    cont.continue_merkle_authority(**args, adapters=ads)
    later2 = LATER + timedelta(minutes=5)
    run161c, wout3, aout3 = _advance_run163(
        base['f'],
        tmp_path / 'advance3',
        seq=3,
        run161_prev=base['run161b'],
        wout_prev=base['wout2'],
        aout_prev=base['aout2'],
        now=later2,
    )
    for _, la, ga in ads:
        la.now = later2
        ga.now = later2
    out2 = tmp_path / 'out2'; args2 = _kwargs(base, out2, run161=run161c, wout=wout3, aout=aout3, now=later2); args2.pop('historical', None)
    res = cont.continue_merkle_authority(**args2, adapters=ads, previous_output_dir=out1)
    assert res['sequence'] == 3 and res['post_handoff_sequence'] == 2
    rec = json.loads((out2/'release-archive-merkle-continuity-receipt.json').read_text())
    assert all(rec['events'][1]['logResponses'][lid]['signed']['consistencyProof'] for lid in rec['events'][1]['logResponses'])
    assert cont.verify_merkle_authority_continuity(**_kwargs(base, out2, run161=run161c, wout=wout3, aout=aout3, now=later2))['ok']


def test_run166_authority_change_after_bridge_requires_new_bridge(base, tmp_path):
    ads = _adapters(base)
    out1 = tmp_path / 'out1'
    args = _kwargs(base, out1)
    args.pop('historical', None)
    cont.continue_merkle_authority(**args, adapters=ads)
    later2 = LATER + timedelta(minutes=5); run161c, wout3, aout3 = _advance_run163(base['f'], tmp_path/'advance3', seq=3, run161_prev=base['run161b'], wout_prev=base['wout2'], aout_prev=base['aout2'], now=later2)
    r165, lid, logs, goss = _rotated_run165(base, tmp_path/'rotation')
    amap = json.loads((r165/'active-archive-log-authority.json').read_text())['authority']
    ads2 = _adapters(base, amap, logs, goss, now=later2)
    args2 = _kwargs(base, tmp_path/'out2', run161=run161c, wout=wout3, aout=aout3, run165=r165, now=later2)
    args2.pop('historical', None)
    with pytest.raises(cont.ArchiveMerkleContinuityError, match='AUTHORITY_CHANGED_REQUIRES_NEW_BRIDGE'):
        cont.continue_merkle_authority(**args2, adapters=ads2, previous_output_dir=out1)


def test_run166_active_authority_cannot_overlap_witness_plane(base, tmp_path):
    stp = base['run165'] / 'trusted-archive-log-authority-state.json'; st = json.loads(stp.read_text()); cur = copy.deepcopy(st['activeAuthority']); nxt = copy.deepcopy(cur); lid = sorted(nxt)[0]
    lp = Ed25519PrivateKey.generate(); gp = Ed25519PrivateKey.generate(); nxt[lid]['publicKey'] = _pub(lp); nxt[lid]['gossipPublicKey'] = _pub(gp)
    wit_key = next(iter(base['f']['wr']['signed']['keys'].values())); nxt[lid]['operator'] = wit_key['operator']
    tr, info = t165._transition(base, tmp_path, sequence=2, kind='scheduled-rotation', current=cur, nxt=nxt, previous_state_path=stp, issued=NOW + timedelta(seconds=2)); logs = dict(base['f']['lpriv']); goss = dict(base['f']['gpriv']); logs[lid] = lp; goss[lid] = gp
    hf = t165._handoff(base, tr, info, new_privs={'log': logs, 'gossip': goss}, old_privs={'log': base['f']['lpriv'], 'gossip': base['f']['gpriv']}); tp = _write(tmp_path / 'tr.json', tr); hp = _write(tmp_path / 'hf.json', hf); r165 = tmp_path / 'run165'
    auth.apply_log_authority_transition(run164_dir=base['run164'], transparency_root_path=base['f']['rp'], transparency_root_pin=base['f']['rpin'], governance_root_path=base['gp'], governance_root_pin=base['gpin'], recovery_root_path=base['rp'], recovery_root_pin=base['rpin'], transition_path=tp, handoff_path=hp, previous_output_dir=base['run165'], output_dir=r165, now=NOW + timedelta(seconds=2))
    amap = json.loads((r165 / 'active-archive-log-authority.json').read_text())['authority']; args = _kwargs(base, tmp_path / 'out', run165=r165); args.pop('historical', None)
    with pytest.raises(cont.ArchiveMerkleContinuityError, match='EXTERNAL_AUTHORITY_OVERLAP'):
        cont.continue_merkle_authority(**args, adapters=_adapters(base, amap, logs, goss))


def test_run166_authority_input_drift_detected_before_commit(base, tmp_path):
    import shutil
    gp = tmp_path/'gov.json'
    shutil.copy2(base['gp'], gp)
    ads = _adapters(base)
    lid, la, ga = ads[0]

    class MutatingLog:
        def __call__(self, req):
            doc = la(req)
            gp.write_bytes(gp.read_bytes() + b' ')
            return doc

    ads[0] = (lid, MutatingLog(), ga)
    out = tmp_path/'out'
    args = _kwargs(base, out)
    args.pop('historical', None)
    args['governance_root_path'] = gp
    with pytest.raises(cont.ArchiveMerkleContinuityError, match='INPUT_DRIFT'):
        cont.continue_merkle_authority(**args, adapters=ads)
    assert not out.exists()
