#!/usr/bin/env bash

# ONE TIME: create the volume: `modal volume create forge_repo`

# Clean-up previous code from the volume
modal volume rm forge_repo data -r
modal volume rm forge_repo forge -r
modal volume rm forge_repo scripts -r

# Add the latest code to the volume
modal volume put forge_repo ./data/configs/ ./data/configs/ --force
modal volume put forge_repo ./forge ./forge --force
modal volume put forge_repo ./scripts ./scripts --force