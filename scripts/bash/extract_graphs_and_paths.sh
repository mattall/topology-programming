
# Run this from the top-level of a results dir.
# Walks all results and extracts the .gml and paths files for all of the solutions from Doppler. 
find . -type f \( -name '*sol_*.gml' -o -path '*/sol*/paths/*' \) -print0 | tar --null -czvf graphs.tar.gz -T -
find data/archive/doppler-1-25-24/results/ \( -name "*sol*.gml" -o -path "*sol*/paths/*" \) -exec tar czvf graphs_and_paths.tar.gz --transform 's#^.*/\([^/]*\)/\([^/]*\)$#\1/\2#' {} +

find data/archive/doppler-1-25-24/results/ \( -name "*sol*.gml" -o -path "*sol*/paths/*"     \) -exec tar czvf graphs_and_paths.tar.gz --transform 's#^.*/\([^/]*\)/\([^/]*\)$#\1/\2#' {} +
find data/archive/doppler-1-25-24/results/ \( -name "*sol*.gml" -o -path "*sol*/paths/mcf_0" \) -exec tar czvf graphs_and_paths.tar.gz --transform 's#^.*/\([^/]*\)/\([^/]*\)-[^/]*$#\1/\2/\2#' {} +

find data/archive/doppler-1-25-24/results/ \( -name "*sol*.gml" -o -path "*sol*/paths/mcf_0" \) -exec tar --show-transformed -cvf graphs_and_paths.tar.gz --transform 's#^.*/\([^/]*\)/\([^/]*\)-[^/]*$#\1/\2/\2#' {} +
find data/archive/doppler-1-25-24/results/ \( -name "*sol*.gml" -o -path "*sol*/paths/mcf_0" \) -exec tar --show-transformed -cvf my.tar.gz --transform 's#^.*/\([^/]*\)/\([^/]*\)-[^/]*$#\1/\2/mcf_0#' {} +

data/archive/doppler-1-25-24/results/

Campus_mcf-Doppler-Campus-background-05-1-0-conservative-05_dynamic_doppler_1_Doppler_conservative_05__-mcf_0/Campus_0-0/mcf_0
data/archive/doppler-1-25-24/results/Campus_mcf-Doppler-Campus-background-05-1-0-conservative-05_dynamic_doppler_1_Doppler_conservative_05__-mcf_0/Campus_0-0-1_1_0_0_0_Gbps_05sol_0/paths/mcf_0

Campus_mcf-Doppler-Campus-background-05-1-0-conservative-05_dynamic_doppler_1_Doppler_conservative_05__-mcf_0_Campus_0-0-1_1_0_0_0_Gbps_05sol_0/mcf_0
Campus_mcf-Doppler-Campus-background-05-1-0-conservative-05_dynamic_doppler_1_Doppler_conservative_05__-mcf_0_Campus_0-0-1_1_0_0_0_Gbps_05sol_0/Campus_0-0-1_1_0_0_0_Gbps_05sol_0.gml

find data/archive/doppler-1-25-24/results/ \( -name "*sol*.gml" -o -path "*sol*/paths/mcf_0" \) -exec tar --show-transformed -cvf my.tar.gz --transform 's#^.*/\([^/]*\)/\([^/]*-[^/]*-[^/]*-[^/]*-[^/]*-[^/]*-[^/]*-[^/]*\)mcf_0/[^/]*$#\1/\2/\2.gml#' {} +

tar: path/to/roots/root/a_sol_0.gml: Cannot stat: No such file or directory

on tar -tvf i see
-rw-r--r-- me/me    3249 2024-01-25 20:57 a-long-string/another-string-sol-fdkj.gml
-rw-r--r-- me/me   10599 2024-01-25 20:57 paths/mcf_0

But I want the structure to be
-rw-r--r-- me/me    3249 2024-01-25 20:57 a-long-string-sol-fdkj/sol-fdkj.gml
-rw-r--r-- me/me   10599 2024-01-25 20:57 a-long-string-sol-fdkj/thing.txt

