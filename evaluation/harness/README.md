# Evaluation harness

The harness exposes one production path:

1. `eval_config.py` loads `nemotron_cascade2.yaml` with exact section keys and
   validates every serving, search, and grader value.
2. `launch_server.py` selects BF16 or Humming W4A8 target weights and independently
   enables DFlash, using the YAML-selected tensor-parallel SGLang server.
3. `validate_server.py` compares the live server, model metadata, runtime markers,
   and GPU state with the YAML before generation.
4. `run_proof_search.py` loads exactly the IDs in the supplied JSON manifest and
   runs the same proof-pool engine for every problem.
5. `proof_search.py` launches all round candidates together, verifies each valid
   proof immediately while other generations continue, then ranks at the round
   barrier. Every call and proof is checkpointed and all budgets come from YAML.
6. `grade_proofs.py` grades the one selected proof per problem for the configured
   64 attempts and applies zero-veto aggregation.
7. `run_full_evaluation.py` pins inputs, performs the audits, and writes the final
   machine-readable and Markdown reports.

HTTP calls are issued once. Every raw response is appended and flushed before it
can affect a derived artifact. Existing successful records are resume
checkpoints; existing failed records terminate the resumed run. There are no
alternate prompts, request retries, model fallbacks, proof fallbacks, stub
graders, or synthetic scores.

The proof prompt files are copied byte-for-byte from ycchen's deployed Math-3R
pipeline at commit `bc03a2c71a076990deaad3d712c6889682e12c69`. The local code
uses ycchen's system/user split, XML output contract, and XML candidate bundle,
while the configurable multi-round search schedule follows the approved
Nemotron-Cascade-style evaluation design.
