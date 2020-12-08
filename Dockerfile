FROM nvcr.io/nvidia/pytorch:20.11-py3

RUN cd /
RUN git clone https://github.com/AgentVi/efficientdet-pytorch.git /efficientdet-pytorch
RUN cd /efficientdet-pytorch
WORKDIR /efficientdet-pytorch
RUN pip install -r ./requirements.txt

ENTRYPOINT ["./distributed_train.sh"]
