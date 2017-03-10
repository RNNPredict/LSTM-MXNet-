#!/bin/sh

WKDIR=$(pwd)
if [ ! -d "${WKDIR}/.git" ]; then
  echo 'This script must be executed on local git repository root dir.' 1>&2
  exit 1
fi
if [ -z "$JOB_QUEUE_NAME" ]; then
  echo "Missing environment variable 'JOB_QUEUE_NAME'." 1>&2
  exit 1
fi
if [ -z "$JOB_DEFINITION_NAME" ]; then
  echo "Missing environment variable 'JOB_DEFINITION_NAME'." 1>&2
  exit 1
fi


JOB_NAME="job-`date +%s`"

# aws --region $AWS_DEFAULT_REGION batch register-job-definition \
#   --job-definition-name ${JOB_DEFINITION_NAME} \
#   --container-properties file://production/aws/batch/job-mxnet.json \
#   --type container

JOB_DEFINITION_ARN=$( aws --region $AWS_DEFAULT_REGION batch describe-job-definitions \
  --job-definition-name ${JOB_DEFINITION_NAME} \
  --status ACTIVE \
  | jq -r '.jobDefinitions | max_by(.revision).jobDefinitionArn' \
) && echo ${JOB_DEFINITION_ARN}


if [ -z "$SEQUENCE_LEN" ]; then
  SEQUENCE_LEN=130
fi
if [ -z "$LSTM_LAYERS" ]; then
  LSTM_LAYERS=3
fi
if [ -z "$UNITS_IN_CELL" ]; then
  UNITS_IN_CELL=512
fi
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=32
fi
if [ -z "$LEARNING_EPOCHS" ]; then
  LEARNING_EPOCHS=10
fi
if [ -z "$LEARNING_RATE" ]; then
  LEARNING_RATE=0.01
fi
if [ -z "$GPUS" ]; then
  GPUS=0
fi

aws --region $AWS_DEFAULT_REGION batch submit-job \
  --job-name ${JOB_NAME} --job-queue ${JOB_QUEUE_NAME} \
  --job-definition ${JOB_DEFINITION_ARN} \
  --container-overrides "{\"environment\": [
    { \"name\": \"SEQUENCE_LEN\", \"value\": \"${SEQUENCE_LEN}\"},
    { \"name\": \"LSTM_LAYERS\", \"value\": \"${LSTM_LAYERS}\"},
    { \"name\": \"UNITS_IN_CELL\", \"value\": \"${UNITS_IN_CELL}\"},
    { \"name\": \"BATCH_SIZE\", \"value\": \"${BATCH_SIZE}\"},
    { \"name\": \"LEARNING_EPOCHS\", \"value\": \"${LEARNING_EPOCHS}\"},
    { \"name\": \"LEARNING_RATE\", \"value\": \"${LEARNING_RATE}\"},
    { \"name\": \"GPUS\", \"value\": \"${GPUS}\"}
  ]}"
