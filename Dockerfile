FROM tensorflow/tensorflow:1.15.5-gpu

USER root

RUN apt-get update
RUN apt-get install -y wget

ARG uid_gid=1000:1000

RUN mkdir /home/user
RUN chown $uid_gid /home/user

ENV HOME=/home/user

RUN mkdir /work
RUN chown $uid_gid /work

RUN cd /tmp; wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
RUN sh /tmp/Miniconda3-py39_4.9.2-Linux-x86_64.sh -b && rm /tmp/Miniconda3-py39_4.9.2-Linux-x86_64.sh

ENV PATH=/home/user/miniconda3/bin:$PATH
RUN which conda

RUN apt-get install -y git
RUN cd /work; git clone https://github.com/ngaggion/ChronoRoot.git; ls

RUN apt-get install -y libxi6 libx11-dev libcairo2-dev libxfixes-dev
RUN apt-get install -y libxcursor-dev libxdamage-dev libxext-dev libxrender-dev libxrandr-dev libxss-dev libxtst-dev
RUN apt-get install -y libxcomposite-dev libxinerama-dev libxtst-dev x11-utils

RUN cd /work/ChronoRoot; conda env create -f env.yml; 

RUN conda init bash
RUN source activate ChronoRoot; pip install tensorflow-gpu==1.15; 
RUN source activate ChronoRoot; pip install opencv-python; 
RUN source activate ChronoRoot; pip install git+https://github.com/lucasb-eyer/pydensecrf.git

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1eVVbWqPUjwYCONeUmx-5nq1wyanhXcTh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eVVbWqPUjwYCONeUmx-5nq1wyanhXcTh" -O modelWeights.zip && rm -rf /tmp/cookies.txt;
RUN mv modelWeights.zip /work/ChronoRoot/modelWeights.zip
RUN unzip /work/ChronoRoot/modelWeights.zip -d /work/ChronoRoot