#! /bin/bash

p2p_result=$(curl -s http://<ipaddr_to_cdi_director>:8080/liqid/api/v2/machine | jq -r '.["response"]["data"][0]["p2p"]')

if [ "$p2p_result" == "on" ]; then
    echo "P2P is enabled"
    exit 0
else
    echo "P2P is disabled"
    exit 1
fi
