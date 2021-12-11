# # If you are using RTX 3080, you need to use CUDA 11.1 
# # which requires driver version >= 450.80.02
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# CUDA 10.1 requires driver version >= 418.39
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install apt-utils libopencv-dev cmake git sudo vim software-properties-common screen wget

# # Install nvidia diver 455 if using CUDA 11.1
#RUN apt-get -y install nvidia-driver-455 

#RUN apt-get purge nvidia-*
#RUN add-apt-repository ppa:graphics-drivers/ppa
#RUN apt-get update

#RUN apt-get -y install nvidia-driver-440
RUN pip install matplotlib tqdm yacs Pillow tensorboardx six==1.13.0 wandb scikit-learn opencv-python coloredlogs pandas dill ncls orjson termcolor
RUN echo 'export PYTHONPATH=/workspace/pedestrian_intent_action_detection:$PYTHONPATH' >> ~/.bashrc
# RUN cd pedestrian_intent_action_detection
# RUN python setup.py build develop
# config wandb
# RUN wandb login YOUR_WANDB_KEY


