.. mermaid::

   stateDiagram-v2

       [*] --> Ready
       Ready --> Validating : request received
       Validating --> Rejected : policy or input failure
       Validating --> Building : validation succeeds

       Building --> Publishing : build succeeds
       Building --> Failed : compiler, timeout, or resource failure
       Publishing --> Available : manifest and artifact verified
       Publishing --> Failed : publication or integrity failure

       Available --> Loading : import requested
       Loading --> Active : import succeeds
       Loading --> Quarantined : metadata or artifact rejected

       Failed --> Cleaning : rollback and diagnostics
       Rejected --> Ready : caller corrects request
       Quarantined --> Cleaning : purge invalid entry
       Cleaning --> Ready : staging and locks released
       Active --> Available : runtime reference released
