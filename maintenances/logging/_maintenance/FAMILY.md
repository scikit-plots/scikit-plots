# Family boundaries

`scikitplot.logging` owns the public logging package facade and private `_logging` implementation.

`scikitplot._cli.logging` separately owns CLI verbosity mapping and stderr policy. It shares the named `scikitplot` Logger, so initialization-order behavior is a joint integration boundary.

Python stdlib `logging` owns Logger/Handler semantics. The scikitplot facade may provide compatibility forwarding but must not silently mutate stdlib global/root policy during introspection.
