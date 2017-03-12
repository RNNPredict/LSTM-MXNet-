#!/bin/sh

WKDIR=$(pwd)
if [ ! -d "${WKDIR}/.git" ]; then
  echo 'This script must be executed on local git repository root dir.' 1>&2
  exit 1
fi

pushd data
rm -rf ./*.txt
find . -name "*.pdf" -exec pdftotext -enc ASCII7 {} \;
popd

cat ./data/*.txt | tr '\n' '.' | tr '\r' '.' \
  | sed -e 's/\.\.*/. /g' -e 's/[[:blank:]][[:blank:]]*/ /g' \
  | fold -w 125 -s > src/train/input.txt
