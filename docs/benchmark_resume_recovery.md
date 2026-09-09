# Failed-task recovery and Windows interruption

The old failed-task recovery changed a task's status to `interrupted` but kept
the previous `failed` record JSON on that task. Resume then failed while loading
fresh-process references because the record and authoritative status differed.
The corrected ledger clears the stale task record in the same transaction as
requeueing, archives its full JSON in the recovery event, and preserves the
failed attempt history. Measured records and payloads are unchanged.

On Windows the old launcher's unlimited `Popen.wait()` enters an infinite
`WaitForSingleObject`. Python cannot dispatch its forwarding handler until the
wait returns. The corrected launcher waits in bounded 100 ms intervals so
Ctrl+C can be forwarded as a controller-group CTRL_BREAK event. A live console
regression covers the actual event and a controller checkpoint.

## Existing immutable runs

Do not replace the source of a run already in progress with these controller
changes. Its execution fingerprint binds the ledger implementation, and the
launcher also verifies the original Git commit. Preserve the original checkout
and use `scripts/resume_windows_benchmark.py` from a separate corrected checkout.

The compatibility helper takes `--repository` pointing at the original clean
checkout, `--result-root`, and optionally `--host-profile`. By default it repairs
and verifies without starting calculations. Add `--execute` to resume the three
pillars through the original launcher. Run it with the original controller's
Python environment.

Before launch, the helper acquires each existing ledger's normal run lock and
checks failed attempts. It requires explicit old recovery-event evidence before
accepting an already interrupted task with a stale failed record. It backs up
the database to a sibling result directory ending in `-recovery`, archives each
original task row in an audit event, and clears only failed retry records.
Measured task rows, payload checksums, attempt history and immutable metadata
are verified and preserved. Any unrelated integrity failure rolls back repair.
It also safely requeues new failed tasks before invoking the original recovery
code, so later explicit resumes cannot reproduce the same mismatch.

During execution Ctrl+C goes directly to the controller's process group. The
helper accounts for Windows Python virtual-environment redirector processes;
the group ID is the outer redirector PID. One Ctrl+C requests a checkpoint and
waits for the controller. A second Ctrl+C force-terminates only this launcher's
process tree. Repeating the identical helper command resumes committed work.

The helper does not replace adapters, change timers or thresholds, rewrite
measured results, or relax source/payload verification. Known failed numerical
or memory outcomes remain in the attempt history and may recur on retry.
