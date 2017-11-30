# Makefile for the Docker image
# MAINTAINER: Jakob Karalus

.PHONY: all build container push deploy clean test

TAG ?= v0.2.0
PREFIX ?= krallistic

all: clear push deploy

clear: 
	kubectl delete tfjobs -l name=bigdata-vilnius

container: 
	docker build -t $(PREFIX)/tf-dist-example:$(TAG) .
	docker build -t $(PREFIX)/tf-dist-example .

push: container
	docker push $(PREFIX)/tf-dist-example:$(TAG)
	docker push $(PREFIX)/tf-dist-example:latest

deploy:
	kubectl apply -f tf-operator-job.yaml

