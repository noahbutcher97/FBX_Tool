# Release Checklist

Use this checklist before merging any Codex-assisted change. `main` is protected by the `Protect main` ruleset and requires a PR plus the `SDK-free workflow checks` status.

## Prepare

1. Start from a clean tree: `git status --branch --short`.
2. Sync `main`: `git switch main` then `git pull --ff-only origin main`.
3. Create a scoped branch, for example `git switch -c chore/update-docs`.
4. Keep unrelated user changes out of the branch.

## Verify Locally

1. Run the focused command that proves the change.
2. For ordinary code or docs, run:

   ```powershell
   .\scripts\verify-fast.ps1 -PytestTarget 'tests/unit'
   ```

3. For Python, script, CI, or hook changes, run:

   ```powershell
   .\scripts\verify-fast.ps1 -IncludeStyle -PytestTarget 'tests/unit'
   .\.fbxenv\Scripts\python.exe -m pre_commit run --all-files
   ```

4. Record skipped checks with the exact reason.

## Open the PR

1. Commit with a short imperative subject, for example `docs: add release checklist`.
2. Push with tracking: `git push -u origin <branch>`.
3. Open a draft PR with the summary, risk, and validation commands.
4. Confirm GitHub reports `SDK-free workflow checks` as successful.

## Merge and Clean Up

1. Merge only after required checks are green.
2. Delete the remote branch after merge.
3. Locally run:

   ```powershell
   git switch main
   git pull --ff-only origin main
   git branch -d <branch>
   git fetch --prune origin
   ```

4. Finish on clean `main` and report the merge commit or PR number.
