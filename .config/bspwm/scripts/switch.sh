#!/bin/bash

x=$(bspc query -N -n focused.fullscreen)
if [ "$x" == '' ]; then
  bspc node -f next.local.window
else 
  bspc node -f next.local.window
  bspc node -t \~fullscreen
fi
