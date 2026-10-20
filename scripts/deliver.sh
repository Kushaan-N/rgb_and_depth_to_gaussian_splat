#!/bin/bash
# Hand a PLY to the user: print the scp command, byte size and md5 (the user verifies downloads against both).
# A pipeline-frame PLY with a sibling *_refaxes.ply is delivered as the refaxes file (gaussians_colmap axes).
#   bash scripts/deliver.sh <ply> [<ply> ...]          (login-node safe: md5 of a few hundred MB)
HOST="${DELIVER_HOST:-${USER}@unity.rc.umass.edu}"
for p in "$@"; do
  p="$(realpath "$p")"
  case "$p" in *_refaxes.ply) ;; *) r="${p%.ply}_refaxes.ply"; [ -f "$r" ] && p="$r" ;; esac
  [ -f "$p" ] || { echo "missing: $p" >&2; continue; }
  echo "scp $HOST:$p ."
  echo "  size $(stat -c %s "$p" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta') bytes, md5 $(md5sum "$p" | cut -d' ' -f1)"
done
