#!/bin/sh

log() {
  text="$(date "+%Y-%m-%d %H:%M:%S") $*"
  echo "$text"
}

log '执行任务'
