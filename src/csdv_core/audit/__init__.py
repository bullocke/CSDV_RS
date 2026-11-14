"""csdv_core.audit: a blind audit of a label run by human interpreters.

``draw`` takes a seeded sample of judged candidates and of complete windows
from a label run, ``build`` turns it into a static page with every image in
place, ``server`` serves that page and stores the answers, and ``score``
turns the answers into rates. The page holds no model answer, no detection
interval and no candidate ID, so the interpreter judges the imagery alone.
"""
