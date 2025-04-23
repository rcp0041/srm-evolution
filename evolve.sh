#!/bin/sh

ding() {
	notify-send "Evolution complete." & macsound quack
}

case "$1" in
	so) ./so-evolution.py 2>/dev/null | tail -1 && ding ;;
	mo) ./mo-evolution.py 2>/dev/null | tail -1 && ding ;;
esac
