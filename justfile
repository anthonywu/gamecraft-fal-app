build:
    docker build \
        --platform linux/amd64 \
        -t anthonywu-gamecraft:latest \
        .

run-sh:
    docker run -it anthonywu-gamecraft:latest /bin/bash

run:
    ruff check gamecraft.py
    fal run gamecraft.py::FalGamecraftModel

deploy:
    fal deploy gamecraft.py::FalGamecraftModel --app-name gamecraft --auth public --strategy recreate --output json
