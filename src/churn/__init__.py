"""Cost-sensitive churn on Cinelume: who gets a retention offer, decided in money.

Role in the pipeline
--------------------
The package. Stages, in the order data flows through them::

    ingest -> staging -> features -> model -> evaluate      (pipeline runs these)
    figures, dashboard                                       (read reports/ only)
    validation                                               (post-hoc, answer key)

``config`` holds every declared assumption, ``economics`` the decision rule,
``guard`` the lock that keeps the simulation's answer key out of training.
Receives: nothing. Returns: nothing. Depends on: nothing. See
docs/ARCHITECTURE.md for the diagrams and a subscriber's path end to end.
"""
