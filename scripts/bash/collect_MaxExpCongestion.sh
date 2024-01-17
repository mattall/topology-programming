# get lines that start with ecmp, spit out 3'rd column (where the value is).
cat data/results/*/*sol_*/MaxExpCongestion* | grep '^ecmp' | awk '{print $3}' 