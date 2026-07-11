# Agentic evaluation harness

The harness deliberately exposes one production evaluation route.

1. `validate_bf16_dflash_server.py` rejects a live server unless DFlash and the
   required numerical configuration are active.
2. `make_batches.py` splits each 30-problem subset into six ordered
   five-problem ID files.
3. `run_agentic_eval.py` reuses `distill_gen/math_3r` to execute six provers,
   two verifications per valid proof, three refiners, and four selectors.
4. `merge_agentic_shards.py` requires exactly the 60 benchmark IDs and copies
   their full stage traces into one run.
5. `agentic_to_responses.py` creates selected, prover, and refined grader views
   from the same saved generation.
6. `grade_proofs.py` sends each non-empty candidate to the required DeepSeek
   grader twice and writes every response before producing summaries.

Failures are terminal. HTTP requests are issued once. Invalid prover, refiner,
selector, or grader output raises instead of selecting another execution path or
inventing a score.

Generation writes one complete JSON trace per problem before appending its slim
`records.jsonl` entry. Grading appends and flushes one lossless record per
candidate and pass. Existing records are resume checkpoints, not retries: the
same completed API call is never repeated.

The correctness and cache-reuse test procedure is documented separately in
