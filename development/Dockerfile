FROM pottava/mxnet:0.9

RUN apt-get update && apt-get autoremove -y \
    && apt-get install -y bash-completion vim curl git \
    && echo ':syntax enable' > /root/.vimrc \
    && apt-get clean
