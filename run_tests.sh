#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_vision unifyai/ivy-vision:latest python3 -m pytest ivy_vision_tests/
