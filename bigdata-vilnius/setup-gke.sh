#!/usr/bin/env bash

gcloud config set compute/zone europe-west1-d


gcloud container clusters create tf-operator-test-cluster \
    --num-nodes 3 \
    --machine-type n1-standard-4 \
    --scopes storage-rw \
    --preemptible \
    --cluster-version=1.8.3-gke.0 \
    --no-async \
    --enable-kubernetes-alpha

gcloud container clusters get-credentials tf-operator-test-cluster

