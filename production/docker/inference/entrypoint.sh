#!/bin/bash

if [ -z "$AWS_DEFAULT_REGION" ]; then
  echo "Missing environment variable 'AWS_DEFAULT_REGION'." 1>&2
  exit 1
fi
if [ -z "$AWS_S3_BUCKET" ]; then
  echo "Missing environment variable 'AWS_S3_BUCKET'." 1>&2
  exit 1
fi
if [ -z "$AWS_S3_PREFIX" ]; then
  echo "Missing environment variable 'AWS_S3_PREFIX'." 1>&2
  exit 1
fi
if [ -z "$LEARNING_EPOCHS" ]; then
  echo "Missing environment variable 'LEARNING_EPOCHS'." 1>&2
  exit 1
fi

export VOCABULARY_FILE="./input.txt"
export PARAMETERS_FILE="model"
AWS_S3_KEY_VOCAB="input.txt"
EPOCHS=$( printf "%04d" ${LEARNING_EPOCHS} )

cd /app

aws --region $AWS_DEFAULT_REGION s3api get-object \
    --bucket $AWS_S3_BUCKET \
    --key $AWS_S3_KEY_VOCAB $AWS_S3_KEY_VOCAB >/dev/null

rc=$?
if [ $rc -ne 0 ] ;then
  echo "[Error] Receiving vocabluary file went wrong..." 1>&2
  echo "AWS_S3_BUCKET: "${AWS_S3_BUCKET} 1>&2
  echo "AWS_S3_KEY_VOCAB: "${AWS_S3_KEY_VOCAB} 1>&2
  exit $rc
fi

aws --region $AWS_DEFAULT_REGION s3api get-object \
    --bucket $AWS_S3_BUCKET \
    --key "${AWS_S3_PREFIX}-${PARAMETERS_FILE}-${EPOCHS}.params" \
    "${PARAMETERS_FILE}-${EPOCHS}.params" >/dev/null

rc=$?
if [ $rc -ne 0 ] ;then
  echo "[Error] Receiving job parameters went wrong..." 1>&2
  echo "AWS_S3_BUCKET: "${AWS_S3_BUCKET} 1>&2
  echo "AWS_S3_KEY: ${AWS_S3_PREFIX}-${PARAMETERS_FILE}-${EPOCHS}.params" 1>&2
  exit $rc
fi

aws --region $AWS_DEFAULT_REGION s3api get-object \
    --bucket $AWS_S3_BUCKET \
    --key "${AWS_S3_PREFIX}-${PARAMETERS_FILE}-symbol.json" \
    "${PARAMETERS_FILE}-symbol.json" >/dev/null

rc=$?
if [ $rc -ne 0 ] ;then
  echo "[Error] Receiving job parameters (symbol) went wrong..." 1>&2
  echo "AWS_S3_BUCKET: "${AWS_S3_BUCKET} 1>&2
  echo "AWS_S3_KEY: ${AWS_S3_PREFIX}-${PARAMETERS_FILE}-symbol.json" 1>&2
  exit $rc
fi

python inference.py

rc=$?
if [ $rc -ne 0 ] ;then
  echo "[Error] Executing batch went wrong..." 1>&2
  exit $rc
fi
