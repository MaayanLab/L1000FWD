FROM gcr.io/tensorflow/tensorflow:0.12.1

EXPOSE 6006

CMD tensorboard --logdir=$LOG_DIR --host 0.0.0.0
