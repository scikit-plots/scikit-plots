.. mermaid::

   flowchart LR

       A[User selects template] --> B[Templates API]
       B --> C[Resolve template family]
       C --> D[Locate package resource]
       D --> E[Load template metadata]
       E --> F[Validate availability and compatibility]
       F --> G[Copy or render template]
       G --> H[Pass generated source to builder]
       H --> I[Compile and return result]
