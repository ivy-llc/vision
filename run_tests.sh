#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_vision ivydl/ivy-vision:latest python3 -m pytest ivy_vision_tests/
