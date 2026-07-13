# Async proof-search validation

## Configuration

- Source commit: `18f3bf6ea4ec0aa96e0902b695b3a71e47be8060`
- Target: BF16 `opd-32b-deploy`
- Draft: BF16 DFlash `dflash-32b-draft-v2test-phaseL`
- Serving: FA3 deterministic, TP1 x DP8
- Client proof-search concurrency: 64 cluster-wide
- SGLang running-request cap: 32 per DP replica

## Outcome

- Status: complete, with no terminal error
- Rounds: 2
- Cumulative proof pool: 64
- Logical calls: 1,088
- Physical requests: 1,103
- Selected proof: `r02-p0015`
- Mean verifier score: 1.0 from 16 valid votes
- External grading: 64/64 attempts scored 7/7; zero veto was not triggered

Both rounds admitted all 32 generated proofs. Each round completed 512 accepted
verifier calls with no malformed verifier output. Eleven round-one generations
and four round-two refinements used the configured continuation. No verifier
needed a continuation.

## Overlap

Verification began while initial proof generation was still running. The first
proof completed in 202.4 seconds, and some of its verifier calls completed about
30 seconds later while 27 initial generations remained in flight.

At the first checkpoint after all 32 round-one generations completed, 198 of 512
verifiers had already completed. At the corresponding round-two checkpoint, 355
of 512 verifiers had completed. Ranking, early stopping, and the next round still
waited for every current-round candidate pipeline.

## Latency

| Measurement | Previous sequential run | Async concurrency-64 run |
|---|---:|---:|
| Round 1 generate plus verify | 41m 26s | 25m 27s |
| Round 2 refine plus verify | 39m 50s | 19m 54s |
| Full evaluation | 1h 22m 04s | 46m 20s |

The full evaluation was 35m 44s faster, a 43.5% wall-time reduction. The local
proof-search portion completed in 45m 22s; external grading and finalization took
59 seconds.

## Server pressure

- Maximum running requests by DP: 11, 12, 12, 10, 13, 12, 11, 12
- Maximum full-KV use: 309,658 / 544,428 tokens (57%)
- Maximum SWA use: 72,930 / 108,885 tokens (67%)
- Maximum queue depth: 0
- Request retractions: 0
