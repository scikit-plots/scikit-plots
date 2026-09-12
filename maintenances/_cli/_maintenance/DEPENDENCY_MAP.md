# Dependency map

```text
shell / console script
        |
        v
scikitplot._cli.app
        |
        +--> argparse frontend (default, stdlib)
        |       \
        |        +--> one neutral registry/IR --> lazy native handlers
        |
        +--> click frontend (optional)
                /

neutral registry --delegate--> scikitplot.mcp.__main__:main
                              (MCP owns argv parsing after handoff)

native handlers --lazy--> project config/utils/version APIs
```

Corpus and Annoy are not direct `_cli` implementation dependencies. If future
commands expose them, prefer delegation or thin lazy adapters and keep ownership
with the target subsystem.
