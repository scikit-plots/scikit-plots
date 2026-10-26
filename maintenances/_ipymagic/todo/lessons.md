# Lessons: `scikitplot._ipymagic`

## Active Rules

### Rule IPM-L1: Tests - start the IPython test shell once, then ask `globalipapp` for it
- **When:** a test needs an IPython shell.
- **Then:** call `globalipapp.start_ipython()` and then
  `globalipapp.get_ipython()` through one helper; never keep the return of
  `start_ipython()` alone.
- **Verified by:** the `get_ipython` helper and `ip` fixture of `test_ipymagic_magics.py`.
- **Added:** 2026-10-26. Root cause: `start_ipython()` returns the shell only
  on its first call (None afterwards), so a second test module got None.

### Rule IPM-L2: Subprocesses - close every pipe in `finally`
- **When:** a `Popen` has `stdin`, `stdout` or `stderr` pipes.
- **Then:** close them in a `finally` after the reader threads end, also on
  timeout and interrupt; a long-running server closes its stdout in `stop()`.
- **Verified by:** the suite under `-W error` (no `ResourceWarning`).
- **Added:** 2026-10-26. Root cause: `Popen` objects that were waited on but
  not used as a context manager left the pipe objects to the garbage collector.

### Rule IPM-L3: Profiling tests - assert order on counts, not on times
- **When:** a test checks which function a profile ranks first.
- **Then:** make the winner strict on `ncalls` (Rule 17); for time keys check
  only that the rows are sorted.
- **Verified by:** `TestProfile.test_table` and `test_sorting_is_deterministic`.
- **Added:** 2026-10-26. Root cause: `tottime` of two cheap functions swaps
  on a busy machine.

### Rule IPM-L4: Server startup - tell "exited" from "too slow"
- **When:** waiting for a child server to print its address.
- **Then:** after the deadline, check `poll()`; report "exited with status N"
  with its output, or "printed no address within T s".
- **Verified by:** `test_failure_is_reported_and_stopped`.
- **Added:** 2026-10-26. Root cause: the first message said "did not start
  within 20 s" for a server that had died at once, hiding its error.

### Rule IPM-L5: Kernel tests - give the kernel the package under test
- **When:** a test starts a Jupyter kernel (a new process).
- **Then:** put the directory holding the imported `scikitplot` on the
  kernel's `PYTHONPATH` (`monkeypatch.setenv`), derived from
  `scikitplot.__file__`, never hard-coded.
- **Verified by:** `test_real_kernel_runs_every_magic` without `PYTHONPATH`.
- **Added:** 2026-10-26. Root cause: from a source checkout the kernel
  imported nothing (`ModuleNotFoundError`) although the test process could.

### Rule IPM-L6: Subprocesses - a pipe is closed by the thread that reads it
- **When:** reader threads call `readline` on a child's pipes.
- **Then:** each reader closes its own pipe when it ends; the waiting thread
  never closes a pipe a reader may hold, and stops forwarding (an `Event`)
  instead of joining without limit. A child that has exited still gets its
  process group signalled (processes it left running).
- **Verified by:** `test_exited_command_is_not_timed_out_and_leftovers_stop`,
  `test_process_in_its_own_session_does_not_hang_the_cell`.
- **Added:** 2026-10-26 (independent review). Root cause: `close()` waits for
  the stream lock that a blocked `readline` holds, so a process that outlived
  the command hung the cell; `_stop` returned early for an exited child, so
  its group was never signalled and a finished command was reported as timed out.

### Rule IPM-L7: Routing - reuse the CLI's own rule, never a parallel grammar
- **When:** the magic decides where a command runs from its arguments.
- **Then:** decide "delegated" as the CLI does (`_split_delegated`: the first
  word anywhere that names a command); test it against that function.
- **Verified by:** `test_route_delegates_as_the_cli_does`.
- **Added:** 2026-10-26 (independent review). Root cause: a regex of root
  flags missed `-vq` and `--verb`, so `env`/`mcp`/`cleanprompt` ran in the kernel.

### Rule IPM-L8: In-process runs - a boundary that turns Ctrl-C into a status hides it
- **When:** calling a `main()` that catches `KeyboardInterrupt`.
- **Then:** record SIGINT around the call (only with Python's default handler,
  main thread) and re-raise after it returns.
- **Verified by:** `TestInProcess.test_interrupt_is_re_raised`.
- **Added:** 2026-10-26 (independent review). Root cause: `_cli.app.main`
  returns 130 on interrupt, right for a process, wrong for "Run all".

### Rule IPM-L9: Deprecations users must see - FutureWarning
- **When:** deprecating a name end users type (`%load_ext ...`).
- **Then:** warn with `FutureWarning`; `DeprecationWarning` is hidden unless
  attributed to `__main__`.
- **Verified by:** `test_deprecation_is_visible_under_default_filters`.
- **Added:** 2026-10-26 (independent review).
