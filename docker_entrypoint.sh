#!/bin/bash
# docker_entrypoint.sh

chmod -R 777 /home/appuser/reemission/outputs
chmod -R 777 /home/appuser/reemission/examples

# If no arguments are passed, run the default command (reemission)
if [ -z "$1" ]; then
  exec reemission "$@"
else
  # Otherwise, run the provided command
  exec "$@"
fi

