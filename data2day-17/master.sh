sudo su
apt-get update && apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg |  apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y kubelet kubeadm

#curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl
#chmod +x ./kubectl
#mv ./kubectl /usr/local/bin/kubectl

for file in /etc/systemd/system/kubelet.service.d/*-kubeadm.conf
do
    echo "Found ${file}"
    FILE_NAME=$file
done

echo "Chosen ${FILE_NAME} as kubeadm.conf"
sed -i '/^ExecStart=\/usr\/bin\/kubelet/ s/$/ --feature-gates="Accelerators=true"/' ${FILE_NAME}

systemctl daemon-reload
systemctl restart kubelet

kubeadm init

exit 
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

source <(kubectl completion bash)

kubectl taint nodes --all node-role.kubernetes.io/master-

export kubever=$(kubectl version | base64 | tr -d '\n')
kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$kubever"