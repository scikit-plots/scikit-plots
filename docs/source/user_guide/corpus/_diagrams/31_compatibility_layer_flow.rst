:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

flowchart TB
    P[Supported Python versions]
    C[_compat compatibility shims]
    E[_StrEnumBase and version helpers]
    S[_schema canonical contracts]
    B[_base component contracts]
    T[_types compatibility records]
    A[Consistent public behavior]

    P --> C
    C --> E
    E --> S
    E --> B
    S --> A
    B --> A
    T -. legacy compatibility .-> A
