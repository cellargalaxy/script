#!/usr/bin/env sh

socat TCP-LISTEN:4747,reuseaddr,fork SYSTEM:"/survive_monitor.sh"