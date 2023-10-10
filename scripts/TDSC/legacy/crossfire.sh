# for network in bellCanada surfNet; do
for network in bellCanada surfNet; do
    for routing in -ecmp -mcf; do 
        for experiment in optimal baseline; do 
            for attack in 100E9 200E9; do
                python /home/mhall/network_stability_sim/eval_scripts/crossfire_every_node_eval.py ${network} ${routing} ${experiment} ${attack} &
            done
        done
    done
done