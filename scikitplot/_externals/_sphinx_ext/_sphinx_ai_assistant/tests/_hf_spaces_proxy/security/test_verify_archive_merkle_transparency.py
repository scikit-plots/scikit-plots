from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT
import base64
import hashlib
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
import test_anchor_archive_health as t163  # type: ignore[import-not-found]
import verify_archive_merkle_transparency as merkle  # type: ignore[import-not-found]

NOW = t163.NOW

def _pub(priv):
    return base64.b64encode(
        priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()


def _sig(priv, obj):
    return base64.b64encode(priv.sign(merkle._canonical(obj))).decode()


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(merkle._canonical(obj))
    return p


def _largest_pow2_lt(n):
    return 1 << ((n - 1).bit_length() - 1)


def _node(a, b):
    return hashlib.sha256(b'\x01' + a + b).digest()


def _mth(leaves):
    if len(leaves) == 1:
        return leaves[0]
    k = _largest_pow2_lt(len(leaves))
    return _node(_mth(leaves[:k]), _mth(leaves[k:]))


def _inclusion(leaves, index):
    if len(leaves) == 1:
        return []
    k = _largest_pow2_lt(len(leaves))
    if index < k:
        return _inclusion(leaves[:k], index) + [_mth(leaves[k:]).hex()]
    return _inclusion(leaves[k:], index - k) + [_mth(leaves[:k]).hex()]


def _subproof(m, leaves, b):
    n = len(leaves)
    if m == n:
        return [] if b else [_mth(leaves).hex()]
    k = _largest_pow2_lt(n)
    if m <= k:
        return _subproof(m, leaves[:k], b) + [_mth(leaves[k:]).hex()]
    return _subproof(m - k, leaves[k:], False) + [_mth(leaves[:k]).hex()]


def _consistency(leaves, old_size):
    if old_size == 0:
        return []
    return _subproof(old_size, leaves, True)


def _root(tmp):
    root_privs = {}
    keys = {}
    for i in range(3):
        kid = f'merkle-root-{i + 1}'
        p = Ed25519PrivateKey.generate()
        root_privs[kid] = p
        keys[kid] = {
            'identity': f'merkle-root-id-{i + 1}',
            'operator': f'merkle-root-op-{i + 1}',
            'expires': merkle._ts(NOW + timedelta(days=800)),
            'publicKey': _pub(p),
        }
    logs = {}
    log_privs = {}
    gossip_privs = {}
    for i in range(3):
        lid = f'merkle-log-{i + 1}'
        lp = Ed25519PrivateKey.generate()
        gp = Ed25519PrivateKey.generate()
        log_privs[lid] = lp
        gossip_privs[lid] = gp
        logs[lid] = {
            'operator': f'merkle-log-op-{i + 1}',
            'publicKey': _pub(lp),
            'gossipIdentity': f'gossip-id-{i + 1}',
            'gossipOperator': f'gossip-op-{i + 1}',
            'gossipPublicKey': _pub(gp),
        }
    selected = ['merkle-root-1', 'merkle-root-2']
    signed = {
        '_type': 'archive-merkle-transparency-root',
        'specVersion': '1.0.0',
        'schemaVersion': 1,
        'rootId': 'archive-merkle/root',
        'version': 1,
        'issuedAt': merkle._ts(NOW),
        'expires': merkle._ts(NOW + timedelta(days=700)),
        'threshold': 2,
        'selectedSignerKeyIds': selected,
        'keys': {k: keys[k] for k in sorted(keys)},
        'logs': {k: logs[k] for k in sorted(logs)},
    }
    doc = {
        'signed': signed,
        'signatures': [
            {'keyId': k, 'signature': _sig(root_privs[k], signed)}
            for k in selected
        ],
    }
    p = _write(tmp / 'merkle-root.json', doc)
    pin = merkle._sha_bytes(merkle._canonical(doc))
    return doc, p, pin, root_privs, log_privs, gossip_privs

class LogAdapter:
    def __init__(self, lid, cfg, priv, now=NOW):
        self.lid = lid
        self.cfg = cfg
        self.priv = priv
        self.now = now
        self.leaves = [
            bytes.fromhex(merkle.merkle_leaf_hash(f'preexisting:{lid}'.encode()))
        ]

    def __call__(self, req):
        assert req['logId'] == self.lid
        old = req['previousTreeSize']
        if old and old != len(self.leaves):
            raise AssertionError((old, len(self.leaves)))
        leaf = bytes.fromhex(req['leafHash'])
        self.leaves.append(leaf)
        n = len(self.leaves)
        idx = n - 1
        root = _mth(self.leaves).hex()
        signed = {
            '_type': 'archive-merkle-log-checkpoint',
            'specVersion': '1.0.0',
            'schemaVersion': 1,
            'logId': self.lid,
            'operator': self.cfg['operator'],
            'sequence': req['sequence'],
            'treeSize': n,
            'rootHash': root,
            'leafIndex': idx,
            'leafHash': req['leafHash'],
            'previousTreeSize': req['previousTreeSize'],
            'previousRootHash': req['previousRootHash'],
            'integratedAt': merkle._ts(self.now),
            'challenge': req['challenge'],
            'inclusionProof': _inclusion(self.leaves, idx),
            'consistencyProof': _consistency(self.leaves, old),
        }
        return {'signed': signed, 'signature': _sig(self.priv, signed)}

class GossipAdapter:
    def __init__(self, lid, cfg, priv, now=NOW, mut=None):
        self.lid = lid
        self.cfg = cfg
        self.priv = priv
        self.now = now
        self.mut = mut

    def __call__(self, req):
        signed = {
            '_type': 'archive-merkle-gossip-observation',
            'specVersion': '1.0.0',
            'schemaVersion': 1,
            'gossipIdentity': self.cfg['gossipIdentity'],
            'gossipOperator': self.cfg['gossipOperator'],
            'sequence': req['sequence'],
            'observedAt': merkle._ts(self.now),
            'challenge': req['challenge'],
            'run163AnchorConsensusHeadSha256': req[
                'run163AnchorConsensusHeadSha256'
            ],
            'checkpointSha256s': dict(req['checkpointSha256s']),
        }
        if self.mut:
            self.mut(signed)
        return {'signed': signed, 'signature': _sig(self.priv, signed)}

def _fixture(tmp, now=NOW):
    r, wr, wrp, wrpin, wpriv, wout, plan, pp, cpriv, opriv, aout, _ = (
        t163._anchored(tmp / 'pred')
    )
    root, rp, rpin, root_privs, lpriv, gpriv = _root(tmp)
    logs = {}
    adapters = []
    for lid, cfg in root['signed']['logs'].items():
        la = LogAdapter(lid, cfg, lpriv[lid], now)
        ga = GossipAdapter(lid, cfg, gpriv[lid], now)
        logs[lid] = la
        adapters.append((lid, la, ga))
    return {
        'r': r,
        'wr': wr,
        'wrp': wrp,
        'wrpin': wrpin,
        'wpriv': wpriv,
        'wout': wout,
        'plan': plan,
        'pp': pp,
        'cpriv': cpriv,
        'opriv': opriv,
        'aout': aout,
        'root': root,
        'rp': rp,
        'rpin': rpin,
        'root_privs': root_privs,
        'lpriv': lpriv,
        'gpriv': gpriv,
        'logs': logs,
        'adapters': adapters,
    }

def _kwargs(f, out, now=NOW):
    r = f['r']
    return dict(
        run160_dir=r['run160'],
        run161_dir=r['out'],
        retention_root_path=r['root_path'],
        retention_root_pin=r['root_pin'],
        witness_root_path=f['wrp'],
        witness_root_pin=f['wrpin'],
        bootstrap_pin=r['setup']['pin'],
        recovery_pin=r['setup']['rr_pin'],
        attestation_pins=[r['setup']['ca_pin']],
        run162_dir=f['wout'],
        anchor_plan_path=f['pp'],
        run163_dir=f['aout'],
        transparency_root_path=f['rp'],
        transparency_root_pin=f['rpin'],
        output_dir=out,
        now=now,
    )

def _anchored(tmp, now=NOW):
    f = _fixture(tmp, now)
    out = tmp / 'run164'
    res = merkle.anchor_merkle_transparency(
        **_kwargs(f, out, now), adapters=f['adapters']
    )
    return f, out, res

def test_run164_merkle_inclusion_and_consistency_known_tree():
    leaves = [
        bytes.fromhex(merkle.merkle_leaf_hash(x))
        for x in (b'a', b'b', b'c', b'd', b'e')
    ]
    root = _mth(leaves).hex()
    for i in range(len(leaves)):
        assert merkle.verify_inclusion_proof(
            leaf_hash=leaves[i].hex(),
            leaf_index=i,
            tree_size=len(leaves),
            proof=_inclusion(leaves, i),
            root_hash=root,
        )
    old = _mth(leaves[:3]).hex()
    assert merkle.verify_consistency_proof(
        old_size=3,
        new_size=5,
        old_root_hash=old,
        new_root_hash=root,
        proof=_consistency(leaves, 3),
    )

def test_run164_domain_separates_leaf_and_node():
    x = hashlib.sha256(b'x').digest()
    assert merkle.merkle_leaf_hash(b'x') != hashlib.sha256(b'x').hexdigest()
    assert merkle._node_hash(x, x) != hashlib.sha256(x + x).digest()

def test_run164_bad_inclusion_rejected():
    leaves = [
        bytes.fromhex(merkle.merkle_leaf_hash(x))
        for x in (b'a', b'b', b'c')
    ]
    proof = _inclusion(leaves, 1)
    proof[0] = '0' * 64
    assert not merkle.verify_inclusion_proof(
        leaf_hash=leaves[1].hex(),
        leaf_index=1,
        tree_size=3,
        proof=proof,
        root_hash=_mth(leaves).hex(),
    )

def test_run164_bad_consistency_rejected():
    leaves = [
        bytes.fromhex(merkle.merkle_leaf_hash(x))
        for x in (b'a', b'b', b'c', b'd')
    ]
    proof = _consistency(leaves, 2)
    proof[-1] = '0' * 64
    assert not merkle.verify_consistency_proof(
        old_size=2,
        new_size=4,
        old_root_hash=_mth(leaves[:2]).hex(),
        new_root_hash=_mth(leaves).hex(),
        proof=proof,
    )

def test_run164_bootstrap_and_offline_verify(tmp_path):
    f, out, res = _anchored(tmp_path)
    assert res['sequence'] == 1
    assert merkle.verify_merkle_history(**_kwargs(f, out))['ok']

def test_run164_wrong_root_pin_rejected(tmp_path):
    f = _fixture(tmp_path)
    kw = _kwargs(f, tmp_path / 'out')
    kw['transparency_root_pin'] = '0' * 64
    with pytest.raises(
        merkle.ArchiveMerkleError, match='PIN_MISMATCH'
    ):
        merkle.anchor_merkle_transparency(**kw, adapters=f['adapters'])

def test_run164_extra_root_signature_rejected(tmp_path):
    f = _fixture(tmp_path)
    d = json.loads(f['rp'].read_text())
    p = Ed25519PrivateKey.generate()
    d['signatures'].append(
        {'keyId': 'extra', 'signature': _sig(p, d['signed'])}
    )
    _write(f['rp'], d)
    f['rpin'] = merkle._sha_bytes(merkle._canonical(d))
    with pytest.raises(
        merkle.ArchiveMerkleError, match='SIGNATURE_SET_INVALID'
    ):
        merkle.anchor_merkle_transparency(
            **_kwargs(f, tmp_path / 'out'), adapters=f['adapters']
        )

def test_run164_external_authority_overlap_rejected(tmp_path):
    f = _fixture(tmp_path)
    d = json.loads(f['rp'].read_text())
    wid = next(iter(f['wr']['signed']['keys']))
    d['signed']['logs']['merkle-log-1']['operator'] = f['wr']['signed']['keys'][wid]['operator']
    d['signatures'] = [
        {'keyId': k, 'signature': _sig(f['root_privs'][k], d['signed'])}
        for k in d['signed']['selectedSignerKeyIds']
    ]
    _write(f['rp'], d)
    f['rpin'] = merkle._sha_bytes(merkle._canonical(d))
    with pytest.raises(merkle.ArchiveMerkleError, match='EXTERNAL_AUTHORITY_OVERLAP'):
        merkle.anchor_merkle_transparency(
            **_kwargs(f, tmp_path / 'out'), adapters=f['adapters']
        )

def test_run164_gossip_split_view_rejected(tmp_path):
    f = _fixture(tmp_path)
    lid = 'merkle-log-3'

    def mut(s):
        s['checkpointSha256s'][next(iter(s['checkpointSha256s']))] = '0' * 64

    f['adapters'] = [
        (
            x,
            la,
            (
                GossipAdapter(
                    x,
                    f['root']['signed']['logs'][x],
                    f['gpriv'][x],
                    NOW,
                    mut,
                )
                if x == lid
                else ga
            ),
        )
        for x, la, ga in f['adapters']
    ]
    with pytest.raises(merkle.ArchiveMerkleError, match='SPLIT_VIEW'):
        merkle.anchor_merkle_transparency(
            **_kwargs(f, tmp_path / 'out'), adapters=f['adapters']
        )


def test_run164_log_signature_mutation_detected(tmp_path):
    f, out, _ = _anchored(tmp_path)
    d = json.loads(
        (out / 'release-archive-merkle-receipt.json').read_text()
    )
    d['events'][0]['logResponses']['merkle-log-1']['signature'] = 'A' * 88
    _write(out / 'release-archive-merkle-receipt.json', d)
    with pytest.raises(merkle.ArchiveMerkleError):
        merkle.verify_merkle_history(**_kwargs(f, out))


def test_run164_gossip_mutation_detected(tmp_path):
    f, out, _ = _anchored(tmp_path)
    d = json.loads(
        (out / 'release-archive-merkle-receipt.json').read_text()
    )
    d['events'][0]['gossipResponses']['merkle-log-1']['signed']['checkpointSha256s'][
        'merkle-log-2'
    ] = '0' * 64
    _write(out / 'release-archive-merkle-receipt.json', d)
    with pytest.raises(merkle.ArchiveMerkleError):
        merkle.verify_merkle_history(**_kwargs(f, out))


def test_run164_run163_document_mutation_detected(tmp_path):
    f, out, _ = _anchored(tmp_path)
    d = json.loads(
        (out / 'release-archive-merkle-receipt.json').read_text()
    )
    d['events'][0]['run163Documents']['active-archive-anchor-evidence.json'][
        'sequence'
    ] = 99
    _write(out / 'release-archive-merkle-receipt.json', d)
    with pytest.raises(merkle.ArchiveMerkleError):
        merkle.verify_merkle_history(**_kwargs(f, out))
def test_run164_duplicate_json_keys_rejected(tmp_path):
    p = tmp_path / 'd.json'
    p.write_text('{"x": 1, "x": 2}\n')
    with pytest.raises(
        merkle.ArchiveMerkleError, match='DUPLICATE_KEY'
    ):
        merkle._read_json(p, 'DUP')


def test_run164_output_exists_rejected(tmp_path):
    f, out, _ = _anchored(tmp_path)
    with pytest.raises(
        merkle.ArchiveMerkleError, match='OUTPUT_EXISTS'
    ):
        merkle.anchor_merkle_transparency(
            **_kwargs(f, out), adapters=f['adapters']
        )


def test_run164_live_checkpoint_survives_adapter_freshness(tmp_path):
    f, out, _ = _anchored(tmp_path)
    assert merkle.verify_merkle_history(
        **_kwargs(f, out, NOW + timedelta(days=5))
    )['ok']


def test_run164_active_checkpoint_eventually_expires(tmp_path):
    f, out, _ = _anchored(tmp_path)
    with pytest.raises(
        merkle.ArchiveMerkleError, match='ACTIVE_EPOCH_STALE'
    ):
        merkle.verify_merkle_history(
            **_kwargs(f, out, NOW + timedelta(days=40))
        )


def test_run164_command_adapter_bounds_output(tmp_path, monkeypatch):
    p = tmp_path / 'noisy.py'
    p.write_text("import sys;sys.stdout.write('x'*(9*1024*1024))")
    real_popen = merkle.subprocess.Popen
    seen = {}

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen['proc'] = proc
        return proc

    monkeypatch.setattr(merkle.subprocess, 'Popen', tracking_popen)
    with pytest.raises(
        merkle.ArchiveMerkleError, match='OUTPUT_TOO_LARGE'
    ):
        merkle.command_log([sys.executable, str(p)])({'x': 1})
    proc = seen['proc']
    assert proc.poll() is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed

def test_run164_second_epoch_proves_real_consistency(tmp_path):
    f = _fixture(tmp_path)
    out1 = tmp_path / 'out1'
    merkle.anchor_merkle_transparency(
        **_kwargs(f, out1), adapters=f['adapters']
    )
    r = f['r']
    later = NOW + timedelta(minutes=5)
    state = json.loads(
        (r['out'] / 'trusted-archive-health-state.json').read_text()
    )
    receipt = json.loads(
        (r['run160'] / 'release-native-evidence-archive-receipt.json').read_text()
    )
    mp2, _ = t163.r162t.r161t._membership(
        tmp_path,
        r['root_doc'],
        r['root_privs'],
        r['run160'],
        receipt,
        r['members'],
        sequence=2,
        previous_head=state['healthChainHeadSha256'],
        previous_members=r['members'],
        issued=later,
        name='membership-2.json',
    )
    targets = []
    for m in r['members']:
        ppv, ap = r['privmap'][m['archiveId']]
        provider = t163.r162t.r161t.Provider(m, ppv, now=later)
        targets.append(
            (
                m['archiveId'],
                provider,
                t163.r162t.r161t.Auditor(m, ap, provider, now=later),
            )
        )
    run161b = tmp_path / 'run161-second'
    t163.r162t.r161t.health.audit_archive_health(
        run160_dir=r['run160'], output_dir=run161b,
        retention_root_path=r['root_path'], membership_path=mp2,
        targets=targets, expected_retention_root_sha256=r['root_pin'],
        expected_bootstrap_root_sha256=r['setup']['pin'],
        expected_recovery_root_sha256=r['setup']['rr_pin'],
        expected_attestation_root_sha256=[r['setup']['ca_pin']],
        previous_output_dir=r['out'], now=later,
    )
    wadapters = [
        (k, t163.r162t.WitnessAdapter(
            k, f['wr']['signed']['keys'][k], f['wpriv'][k], now=later,
        ))
        for k in ['wit-key-1', 'wit-key-2', 'wit-key-3']
    ]
    wout2 = tmp_path / 'witness-second'
    t163.r162t.witness.witness_archive_health(
        run160_dir=r['run160'], run161_dir=run161b,
        retention_root_path=r['root_path'], retention_root_pin=r['root_pin'],
        witness_root_path=f['wrp'], witness_root_pin=f['wrpin'],
        bootstrap_pin=r['setup']['pin'], recovery_pin=r['setup']['rr_pin'],
        attestation_pins=[r['setup']['ca_pin']], output_dir=wout2,
        witnesses=wadapters, previous_output_dir=f['wout'], now=later,
    )
    cadapters = [
        (
            cid, t163.Channel(cid, cfg, f['cpriv'][cid], later),
            t163.Observer(cid, cfg, f['opriv'][cid], later),
        )
        for cid, cfg in f['plan']['signed']['channels'].items()
    ]
    aout2 = tmp_path / 'anchor-second'
    t163.anchor.anchor_archive_health(
        run160_dir=r['run160'], run161_dir=run161b,
        retention_root_path=r['root_path'], retention_root_pin=r['root_pin'],
        witness_root_path=f['wrp'], witness_root_pin=f['wrpin'],
        bootstrap_pin=r['setup']['pin'], recovery_pin=r['setup']['rr_pin'],
        attestation_pins=[r['setup']['ca_pin']], run162_dir=wout2,
        anchor_plan_path=f['pp'], output_dir=aout2, channels=cadapters,
        previous_output_dir=f['aout'], now=later,
    )
    for _, la, ga in f['adapters']:
        la.now = later; ga.now = later
    out2 = tmp_path / 'out2'
    kw = _kwargs(f, out2, later)
    kw.update(run161_dir=run161b, run162_dir=wout2, run163_dir=aout2)
    res = merkle.anchor_merkle_transparency(**kw, adapters=f['adapters'], previous_output_dir=out1)
    assert res['sequence'] == 2
    bundle = json.loads((out2 / 'release-archive-merkle-bundle.json').read_text())
    assert [e['sequence'] for e in bundle['events']] == [1, 2]
    assert all(row['treeSize'] == 3 for row in bundle['events'][1]['logs'])
    receipt2 = json.loads((out2 / 'release-archive-merkle-receipt.json').read_text())
    assert all(receipt2['events'][1]['logResponses'][lid]['signed']['consistencyProof'] for lid in f['root']['signed']['logs'])
    assert merkle.verify_merkle_history(**kw)['ok']

def test_run164_same_run163_epoch_cannot_be_anchored_twice(tmp_path):
    f, out1, _ = _anchored(tmp_path)
    with pytest.raises(merkle.ArchiveMerkleError, match='SEQUENCE_INVALID'):
        merkle.anchor_merkle_transparency(
            **_kwargs(f, tmp_path / 'out2'),
            adapters=f['adapters'],
            previous_output_dir=out1,
        )

def test_run164_authority_input_drift_detected(tmp_path):
    f = _fixture(tmp_path)
    lid = 'merkle-log-1'
    target = f['pp']
    base = f['adapters'][0][1]

    class MutatingLog:
        def __call__(self, req):
            result = base(req)
            target.write_bytes(target.read_bytes() + b'\n')
            return result

    f['adapters'][0] = (lid, MutatingLog(), f['adapters'][0][2])
    with pytest.raises(merkle.ArchiveMerkleError, match='INPUT_DRIFT'):
        merkle.anchor_merkle_transparency(**_kwargs(f, tmp_path / 'out'), adapters=f['adapters'])


def test_run164_documentation_mentions_merkle_inclusion_consistency_and_gossip():
    guide = (SEC / 'RELEASE_ARCHIVE_MERKLE_GUIDE.md').read_text().lower()
    gates = (SEC / 'SECURITY_RELEASE_GATES.md').read_text()
    for term in ('rfc6962', 'inclusion proof', 'consistency proof', 'gossip', 'split view', 'offline'):
        assert term in guide
    assert 'Run 164' in gates
