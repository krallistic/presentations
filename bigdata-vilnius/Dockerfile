FROM gcr.io/tensorflow/tensorflow:1.3.0-rc2
RUN mkdir -p /opt/mlkube
COPY model/tf_model.py /opt/kube/
ENTRYPOINT ["python", "/opt/kube/tf_model.py"]
