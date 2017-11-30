
#Create the network:
gcloud compute networks create kubernetes-gpu --mode custom

#Subnet:
gcloud compute networks subnets create kubernetes \
  --network kubernetes-gpu \
  --range 10.240.0.0/24

#Firewall, Internal Traffik
gcloud compute firewall-rules create kubernetes-gpu-allow-internal \
  --allow tcp,udp,icmp \
  --network kubernetes-gpu \
  --source-ranges 10.240.0.0/24,10.200.0.0/16

#External Trafik
gcloud compute firewall-rules create kubernetes-gpu-allow-external \
  --allow tcp:22,tcp:6443,icmp \
  --network kubernetes-gpu \
  --source-ranges 0.0.0.0/0
#Loadbalancers
gcloud compute firewall-rules create kubernetes-gpu-allow-health-checks \
  --allow tcp:8080 \
  --network kubernetes-gpu \
  --source-ranges 209.85.204.0/22,209.85.152.0/22,35.191.0.0/16

#Public IP
gcloud compute addresses create kubernetes-gpu \
  --region $(gcloud config get-value compute/region)  

# Compute Instances
## Controllers 
for i in 0 1 2; do
  gcloud compute instances create controller-${i} \
    --async \
    --boot-disk-size 200GB \
    --can-ip-forward \
    --image-family ubuntu-1604-lts \
    --image-project ubuntu-os-cloud \
    --machine-type n1-standard-1 \
    --private-network-ip 10.240.0.1${i} \
    --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring \
    --subnet kubernetes \
    --tags kubernetes-gpu,controller
done

## CPU Workers
for i in 0 1 2; do
  gcloud compute instances create worker-${i} \
    --async \
    --boot-disk-size 200GB \
    --can-ip-forward \
    --image-family ubuntu-1604-lts \
    --image-project ubuntu-os-cloud \
    --machine-type n1-standard-1 \
    --metadata pod-cidr=10.200.${i}.0/24 \
    --private-network-ip 10.240.0.2${i} \
    --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring \
    --subnet kubernetes \
    --tags kubernetes-gpu,worker
 done

## GPU Workers 
for i in 0 1 2; do
  gcloud compute instances create worker-${i} \
    --async \
    --boot-disk-size 200GB \
    --can-ip-forward \
    --image-family ubuntu-1604-lts \
    --image-project ubuntu-os-cloud \
    --machine-type n1-standard-1 \
    --metadata pod-cidr=10.200.${i}.0/24 \
    --private-network-ip 10.240.0.2${i} \
    --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring \
    --subnet kubernetes \
    --tags kubernetes-gpu,worker,gpu
 done