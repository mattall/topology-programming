# Full experiment
for network in fishnet sprint ANS CRL bellCanada surfNet
# Short Experiment
# for network in sprint
do
    for c in {10}
    do
        echo "python net_sim.py $network add_circuit -C $c -H"; 
    done
done