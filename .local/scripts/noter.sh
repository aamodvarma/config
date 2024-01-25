#/bin/bash
# noter
f=$HOME/Documents/sb/notes/$(date +"%Y-%m-%d.md") ; { echo; date +"# %H:%M" ; echo ; } >> "$f" ; nvim + "$f" ;
