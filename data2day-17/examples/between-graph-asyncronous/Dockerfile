FROM gcr.io/tensorflow/tensorflow:1.3.0-gpu


ADD MNIST_data /MNIST_data
#Add additional requirements/data
ADD mnist.py /

ENTRYPOINT ["python", "/mnist.py"]


