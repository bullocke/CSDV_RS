"""csdv_core.review — interpret detected disturbances in a browser app.

:mod:`~csdv_core.review.sample` draws the review sample from exported grid
runs, :mod:`~csdv_core.review.store` keeps the labels and redrawn outlines,
:mod:`~csdv_core.review.app` is the Solara app (served with its map tiles by
:mod:`~csdv_core.review.server`) and :mod:`~csdv_core.review.summary` scores
the detections. The ``csdv review`` commands are in
:mod:`~csdv_core.review.cli`. See docs/guides/disturbance_review.md.
"""
