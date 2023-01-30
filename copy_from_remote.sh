IP=$1

if [ "$IP" = "-h" ]; then
    echo "CPU: 20.88.107.10"
    echo "GPU: 20.22.152.67"
    exit 0
fi

if [ "$IP" != "20.88.107.10" ] && [ "$IP" != "20.22.152.67" ]; then
    echo "$IP" is not a valid IP. Aborting...
    exit 1
else
    echo "$IP" is valid. Starting...
    echo "Storing in $(pwd)/remote"
    rsync --progress -vr -e "ssh -p 50000" azureuser@"$IP":/home/azureuser/localfiles/aml22-rl/trained_models remote/
     rsync --progress -vr -e "ssh -p 50000" azureuser@"$IP":/home/azureuser/localfiles/aml22-rl/logs remote/
    exit 0
fi