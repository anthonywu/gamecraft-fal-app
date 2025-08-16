FROM hunyuanvideo/hunyuanvideo:cuda_12

RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0.git /opt/gamecraft

WORKDIR /opt/gamecraft

RUN echo hf_transfer >> ./requirements.txt && \
    pip install -q -r ./requirements.txt && \
    which python && \
    python -V && \
    pip list && \
    ls -ld */*/* /opt/gamecraft

ADD "https://hunyuan-gamecraft.github.io/assets/videos/singleaction/input/13.png" /opt/gamecraft/sample-input-parisian-street.png

ENV PYTHONPATH=/opt/gamecraft
