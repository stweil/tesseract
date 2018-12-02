#!/bin/bash

export LANG=C
export OMP_THREAD_LIMIT=1

i=$1

num=$(basename "$i" .jp2)
d=$(dirname "$i")
film=$(basename "$d")
dir=$film
hocr="$dir/$num.hocr"
txt="$dir/$num.txt"
jp2="$dir/$num.jp2"

base=$PWD

if test ! -s "$hocr"; then
  mkdir -p "$dir"
  echo "$dir/$num"
  (
    cd "$d"
    time -p nice /home/stweil/src/github/tesseract-ocr/tesseract/bin/ndebug/clang/src/api/tesseract "$num.jp2" "$base/$dir/$num" -l Fraktur --tessdata-dir /home/stweil/src/github/tesseract-ocr/tesseract/tessdata hocr
  )
fi
