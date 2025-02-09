#!/usr/bin/env bash
docker build -t ibkr_tws .
docker container stop ibkr_tws
docker container rm ibkr_tws
docker create \
    --name ibkr_tws \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ibkr_tws