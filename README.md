# Anglerfish

## Environment usage

First-time Python setup:
```sh
poetry env use 3.12
```

Then, each time when you need the shell, just run:
```sh
eval $(poetry env activate)
```

Install dependencies:
```sh
poetry install --with=dev --with=test --no-root
```

© 2025 Alexander Myodov (amyodov@gmail.com).
