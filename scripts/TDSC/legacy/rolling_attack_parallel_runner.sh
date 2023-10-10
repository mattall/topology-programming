# for network in bellCanada surfNet; do

for network in ANS CRL bellCanada surfNet; do
    for routing in -ecmp -mcf; do 
        for experiment in optimal baseline; do                 
                echo "screen -dmS ${network}-${routing}-${experiment} sh -c \"source venv/bin/activate; python /home/mhall/network_stability_sim/eval_scripts/rolling_attack_parallel_worker.py ${network} ${routing} ${experiment}\""
        done
    done
done