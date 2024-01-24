te="mcf"
basefile=/home/mhall/topology-programming/data/results/Campus_${te}-Doppler-Campus-background-05_1__-${te}_10/Campus_0-0-1_1_0_0_0_Gbps_05sol_
example=/home/mhall/topology-programming/data/results/Campus_${te}-Doppler-Campus-background-05_1__-${te}_10/Campus_0-0-1_1_0_0_0_Gbps_05sol_0/paths/${te}_0
for f in ${basefile}*/paths/${te}_0; 
do 
echo $f; echo "==================================================="; 
diff $f $example; 
done;



while read t; do
    while read v; do 
        if [[ $t != $v ]]; then
            echo "=============================" >> topo_diff.txt; 
            echo $v >> topo_diff.txt; 
            echo $t >> topo_diff.txt; 
            echo "=============================" >> topo_diff.txt; 
            diff $t $v >>  topo_diff.txt; 
        fi
    done < topos.txt;  
done < topos.txt;  