# Dependency Map — `scikitplot.cython`

Derived from the tree by AST, not asserted.

---

## 1. Position: fully independent

```text
scikitplot.cython        imports NO sibling submodule
   │
   └── stdlib + the toolchain it shells out to (Cython, a C compiler)
```

**Zero antecessors, zero successors.** Verified: an AST pass over every `.py`
file finds no import of any other `scikitplot` submodule.

That makes it the only one of the six that can be reviewed, changed and released
without coordinating with another campaign — and the only one where a mistake
cannot propagate sideways.

## 2. But its dependency is unusual

The other five depend on *code*. This one depends on a **toolchain**:

| Depends on | Kind | When |
|---|---|---|
| Cython | build-time subprocess | at `build()` |
| a C/C++ compiler | build-time subprocess | at `build()` |
| the filesystem | lock, cache, staging | throughout |
| **caller-supplied source** | **input** | at `build()` |

That last row is what makes this submodule different from every other one in the
project. **It compiles and loads code the caller hands it.** Every other
submodule processes data; this one processes *programs*.

## 3. The consequence

A retrieval bug returns a wrong document. A build-service bug **runs the wrong
code**, or runs the right code with the wrong privileges, or leaves a lock held
after the process that took it is gone.

The other five campaigns' governing rule was *never let an operation succeed on
partial evidence*. Here it becomes stricter:

> Never let an operation succeed on **unvalidated input**, and never leave a
> resource in a state the next process cannot reason about.

## 4. Review order

```text
corpus ✅ → mcp M00 → annoy A00 → sphinx S00 → _cli C00 → cython Y00
```

Position in that list is a matter of convenience, not dependency: `cython` has no
edges, so it can be reviewed at any point. It is placed last because the other
five constrain each other and this one constrains nobody.

## 5. What it must not become

It would be easy for another submodule to start importing this one — an
`annoy` specialization compiled on demand, a `corpus` kernel built at first use.
Each would be a real capability and each would create the project's first edge
into a subprocess-invoking, filesystem-locking service.

If that is ever proposed, it needs its own review run. An optional dependency on
*a compiler* is not the same shape as an optional dependency on a library.
