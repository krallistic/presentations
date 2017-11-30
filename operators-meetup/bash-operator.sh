#!/bin/bash
SERVER='localhost:8001'
while true;
do
    for HELLO in $(kubectl --server $SERVER get helloworld -o json | jq -c '.items[].spec')
;
    do
        echo $(echo $HELLO | jq .hello | tr -d '"') $(echo $HELLO | jq .world | tr -d '"') 
        kubectl --server $SERVER create deployment $(echo $HELLO | jq .hello | tr -d '"')$(echo $HELLO | jq .world | tr -d '"') --image=busybox
        #curl --header "Content-Type:application/json" --request POST --data '{"apiVersion":"v1", "kind": "Binding", "metadata": {"name": "'$PODNAME'"}, "target": {"apiVersion": "v1", "kind": "Node", "name": "'$CHOSEN'"}}' http://$SERVER/api/v1/namespaces/default/pods/$PODNAME/binding/
        #echo "Created cronjob $PODNAME to $CHOSEN"
    done
    sleep 10
done