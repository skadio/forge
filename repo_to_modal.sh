#!/usr/bin/env bash

# Create the volume: `modal volume create forge_repo`
modal volume put forge_repo ./data/configs/ ./data/configs/ --force
modal volume put forge_repo ./forge ./forge --force
modal volume put forge_repo ./scripts ./scripts --force