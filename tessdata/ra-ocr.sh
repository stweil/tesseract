#!/bin/bash

i=$1

num=$(basename "$i" .jp2)
d=$(dirname "$i")
film=$(basename "$d")
dir=$film
hocr="$dir/$num.hocr"
txt="$dir/$num.txt"
jp2="$dir/$num.jp2"

if test $HOSTNAME = ub-backup; then
  REMOTE=false
else
  REMOTE=true
fi

if test ! -s "$hocr"; then
  export LANG=C
  export OMP_THREAD_LIMIT=1
  base="$PWD"
  TESSBASE="/home/stweil/src/github/tesseract-ocr/tesseract"
  PATH="$TESSBASE/bin/ndebug/clang/src/api:$PATH"

  mkdir -p "$dir"
  $REMOTE && rsync -a "ub-backup:$(echo $i|perl -pe 's/ /\\ /g')" "$jp2"
  echo "$dir/$num"
  (
    $REMOTE || cd "$d"
    $REMOTE && cd "$dir"
    time -p nice tesseract "$num.jp2" "$base/$dir/$num" -l Fraktur --tessdata-dir "$TESSBASE/tessdata" hocr
  )
  $REMOTE && rm -f "$jp2"
fi
