#!/usr/bin/env bash

kubectl apply -f sa-rbac.yaml
helm init --service-account tiller --tiller-namespace default
CHART=https://storage.googleapis.com/tf-on-k8s-dogfood-releases/latest/tf-job-operator-chart-latest.tgz
sleep 20
kubectl get pods
helm install ${CHART} -n tf-job --wait --replace --set rbac.install=true,cloud=gke --tiller-namespace=default
kubectl get pods
