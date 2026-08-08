# Contributing

Bug reports and focused pull requests are welcome. Please open an issue before a
large API change. Development setup:

```bash
python -m pip install -e ".[test,docs,dev]"
python -m pytest
sphinx-build -W -b html docs docs/_build/html
```

New criteria must document their statistical target and include a numerical
unit test. New pullback metrics must state the observable space and coordinate
transformation convention.
