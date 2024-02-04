# get lines that start with ecmp, spit out 3'rd column (where the value is).
# cat data/results/*/*sol_*/MaxExpCongestion* | grep '^ecmp' | awk '{print $3}' 


Networks=("ANS CRL sprint bellCanada surfNet")
Routings=("mcf" "ecmp")
Scales=("100E9" "200E9")
Iters=("0" "1")

for network in ${Networks[*]}; do
for routing in ${Routings[*]}; do
for scale in ${Scales[*]}; do
for iter in ${Iters[*]}; do
# echo "cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/MaxExpCongestion*"
cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/MaxExpCongestion* | grep ${routing} | awk '{print $3}' >> data/archive/crossfire-rerun-02-01-24/congestion/${network}-${routing}-${scale}-${iter}.dat; 
done
done
done
done