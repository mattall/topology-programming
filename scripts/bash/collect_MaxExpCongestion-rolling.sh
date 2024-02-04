Networks=("ANS CRL sprint bellCanada surfNet")
Routings=("mcf" "ecmp")
Strategies=("baseline" "onset")
for network in ${Networks[*]}; do
for routing in ${Routings[*]}; do
for strat in ${Strategies[*]}; do
echo "${network} ${routing} ${strat}"
cat data/results/${network}_${network}*${routing}*${strat}*/*s_10/MaxCongestionVsIterations.dat | grep ${routing} | awk '{print $3}' >> data/archive/rolling/congestion/
# cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/MaxExpCongestion* | grep ${routing} | awk '{print $3}' >> data/archive/crossfire-rerun-02-01-24/congestion/${network}-${routing}-${scale}-${iter}.dat; 
done
done
done
done
