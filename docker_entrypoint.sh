#!/bin/bash
# docker_entrypoint.sh

# If no arguments are passed, run the default command (reemission)
if [ -z "$1" ]; then
  exec reemission "$@"
else
  # Otherwise, run the provided command
  exec "$@"
fi

