# Agentic evaluation harness

The harness deliberately exposes one production evaluation route.

1. `validate_bf16_dflash_server.py` rejects a live server unless DFlash and the
   required numerical configuration are active.
2. `make_batches.py` splits each 30-problem subset into six ordered
   five-problem ID files.
3. `run_notebook_v2_eval.py` imports the exact hash-pinned notebook scheduler,
   admits 12 total calls, caps prove/refine at 6, prioritizes verifiers, starts
   six provers, uses three verifiers per candidate, and runs five selectors.
4. `merge_agentic_shards.py` requires exactly the 60 benchmark IDs and copies
   their full stage traces into one run.
5. `agentic_to_responses.py` creates selected, prover, and refined grader views
   from the same saved generation.
6. `grade_proofs.py` sends each non-empty candidate to the required DeepSeek
   grader twice and writes every response before producing summaries.

Failures are terminal. HTTP requests are issued once. The wrapper rejects call
errors and every notebook fallback final source instead of accepting them as
evaluation answers.

Generation writes one complete JSON trace per problem before appending its slim
`records.jsonl` entry. Grading appends and flushes one lossless record per
candidate and pass. Existing records are resume checkpoints, not retries: the
same completed API call is never repeated.

The correctness and cache-reuse test procedure is documented separately in
