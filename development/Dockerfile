FROM pottava/mxnet:0.9

RUN apt-get update && apt-get autoremove -y \
    && apt-get install -y bash-completion vim curl git \
    && apt-get install -y python-pip \
    && pip install Flask \
    && echo ':syntax enable' > /root/.vimrc \
    && apt-get clean
