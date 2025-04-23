#!/bin/sh
i=1
case "$#" in
	2) ;;
	*) echo "Usage: produce-series \$script \$N" && exit;;
esac
case "$1" in
	mo) ;;
	so) ;;
	*) echo "Error: must specify 'mo' or 'so'." && exit ;;
esac
while [ "$i" -le "$2" ]; do
	./evolve.sh "$1"
	i=$((i+1))
done
notify-send "Complete!" & macsound pong2003
