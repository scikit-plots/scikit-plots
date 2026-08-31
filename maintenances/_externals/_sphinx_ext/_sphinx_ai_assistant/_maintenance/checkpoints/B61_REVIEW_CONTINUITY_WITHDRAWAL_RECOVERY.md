# B61 — Review Continuity & Portable Withdrawal Recovery

Status: **GREEN CANDIDATE**

## Scope

Close two contributor/reviewer lifecycle gaps in provider-native dataset review:

1. repeated submission of the same logical contribution must update one review rather
   than create reviewer queue spam;
2. closing/reopening the panel or losing browser state must not strand the participant's
   delete/withdraw authority.

## Review continuity

One receipt owns one stable provider review and one stable review file path.

- identical reviewed payload -> no provider commit (`reviewUpdate="unchanged"`);
- changed reviewed payload -> update the existing review (`reviewUpdate="updated"`);
- Hugging Face updates `refs/pr/N`;
- GitHub/GitLab/Bitbucket update the existing source branch;
- the provider review ID is persisted in the receipt lifecycle so normal status/update
  operations use direct lookup instead of scanning large review queues;
- the stable file path is receipt-derived and non-identifying:
  `contributions/YYYY/MM/DD/ct_<review-key>.jsonl`.

## Participant recovery UX

The contribution sheet now exposes multiple recovery paths:

- active tab receipt -> management controls rehydrate automatically when the sheet opens;
- **Save private receipt** -> portable JSON management capability;
- **Copy private withdrawal code** -> compact `aicm2.…` encoding of the same capability;
- **Recover withdrawal access** -> import either private form later;
- **Copy support reference** -> non-secret receipt/provider-review/file locator;
- **Copy maintainer removal request** -> ready-to-send non-secret support text.

Private receipts/codes are never written into provider review metadata or `.jsonl` files.
Normal URL-based withdrawal links are intentionally not used because secrets in URLs can
leak through browser history, logs, referrers, screenshots, or analytics.

## Cross-site safety

Schema-v2 management receipts include a non-authoritative `serviceHint`. Import never
uses that field as a destination. Instead it must match the currently configured
contribution endpoint before the management capability can be sent. This prevents a
receipt copied from one deployment from being accidentally disclosed to an unrelated
fork/service. Schema-v1 receipts remain compatible.

## Maintainer support reference

The proxy returns only bounded non-secret support metadata:

- `reviewProvider`;
- numeric `reviewId`;
- stable `reviewPath`.

It does **not** return `reviewUrl`, repository tokens, raw contribution content, or the
participant delete/withdraw capability. The support reference therefore remains useful
when a saved private capability no longer resolves because pending lifecycle state
expired or a non-durable ledger was lost.

## Durability boundary

A portable receipt/code preserves the participant-held secret but cannot extend the
server's lifecycle retention. Production deployments still need durable SQLite or shared
Redis receipt authority. Provider PR/MR persistence and receipt-lifecycle persistence are
separate concerns.

## Verification

- provider continuity / support metadata / direct-review lookup tests: GREEN;
- contribution privacy/lifecycle/provider/storage/browser focused plane: GREEN;
- JavaScript and Python syntax: GREEN;
- no withdrawal token is introduced into URL, Git metadata, or provider record path;
- support locator explicitly excludes provider review URL and management capability.
