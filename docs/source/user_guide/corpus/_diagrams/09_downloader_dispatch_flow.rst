:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        A[AnyDownloader]
        C[Classify each URL]
        B[Broadcast or align per-URL parameters]
        K{Provider}
        G[GoogleDriveDownloader]
        H[GitHubDownloader]
        Y[YouTubeDownloader]
        W[WebDownloader]
        P[BaseDownloader policy and limits]
        S[Stream to destination]
        R[DownloadResult]

        A --> C
        C --> B
        B --> K
        K --> G
        K --> H
        K --> Y
        K --> W
        G --> P
        H --> P
        Y --> P
        W --> P
        P --> S
        S --> R
