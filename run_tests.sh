#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/vision unifyai/vision:latest python3 -m pytest ivy_vision_tests/
