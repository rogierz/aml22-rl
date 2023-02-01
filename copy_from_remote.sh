IP=$1

if [ "$IP" = "-h" ] || [ "$IP" = "" ]; then
    echo "Usage: $0 [-h] IP"
    echo "Valid IP are:"
    echo "  Francesco:"
    echo "    - CPU: 20.22.91.51" #50001
    echo "  Riccardo:"
    echo "    - CPU: 20.85.61.104" # 50001
    echo "    - GPU: 20.97.157.108" # 50001
    echo "  Ruggero:"
    echo "    - CPU: 20.88.107.10"
    echo "    - GPU: 20.22.152.67"
    
    exit 0
fi

PORT="50000"

if [ "$IP" = "20.22.91.51" ] || [ "$IP" = "20.22.152.67" ] || [ "$IP" = "20.85.61.104" ]; then
    PORT="50001"
fi

if [ "$IP" = "20.88.107.10" ] || [ "$IP" = "20.22.152.67" ] || [ "$IP" = "20.22.91.51" ] || [ "$IP" = "20.85.61.104" ] || [ "$IP" = "20.97.157.108" ]; then
    echo "$IP" is valid. Starting...
    echo "Storing in $(pwd)/remote"
    rsync --progress -vr -e "ssh -p $PORT" azureuser@"$IP":/home/azureuser/localfiles/aml22-rl/trained_models remote/
    rsync --progress -vr -e "ssh -p $PORT" azureuser@"$IP":/home/azureuser/localfiles/aml22-rl/logs remote/
    echo Done!
    exit 0
else
    echo "$IP" is not a valid IP. Aborting...
    exit 1
fi
