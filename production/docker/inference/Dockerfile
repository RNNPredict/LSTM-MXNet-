# MXNet with NVIDIA CUDA 8.0 & cuDNN 5.1
# https://hub.docker.com/r/nvidia/cuda/
# docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mxnet:inference -f production/docker/inference/Dockerfile .

FROM ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mxnet:0.8.0

ADD production/docker/inference/entrypoint.sh /

RUN apt-get update && apt-get autoremove -y \
    && apt-get install -y python-pip \
    && pip install "awscli==1.11.60" \
    && apt-get clean \
    && chmod +x /entrypoint.sh

ADD app/rnn.py /app/
ADD app/rnn_model.py /app/
ADD app/lstm.py /app/
ADD app/inference.py /app/

ENV START_WITH="AWS Batch is" \
    SEQUENCE_LEN=500 \
    LSTM_LAYERS=3 \
    UNITS_IN_CELL=512 \
    LEARNING_EPOCHS=10 \
    LEARNING_RATE=0.01 \
    GPUS=""

ENV AWS_DEFAULT_REGION=us-east-1 \
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_S3_BUCKET
    AWS_S3_PREFIX=${LSTM_LAYERS}-${UNITS_IN_CELL}-${LEARNING_EPOCHS}-${LEARNING_RATE}

CMD ["/entrypoint.sh"]
