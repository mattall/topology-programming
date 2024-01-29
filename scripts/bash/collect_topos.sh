networks=("Campus" "Regional")
tes=("mcf" "ecmp")
ftxs=(1 2 3)
top_ks=(10 20 30 40 50 60 70 80 90 100)
metrics="MaxExpCongestion"

for network in ${networks[*]}; do 
for te in ${tes[*]}; do
for ftx in ${ftxs[*]}; do
for top_k in ${top_ks[*]}; do 
for metric in ${metrics[*]}; do
    cat data/results/${network}_${te}-Doppler-${network}-background-05_${ftx}__-${te}_${top_k}/*sol*/${metric}* | grep "^$te" | awk '{print $3}' > data/reports/${network}_${te}_${ftx}_${top_k}_${metric}.dat
done 
done 
done 
done 
done