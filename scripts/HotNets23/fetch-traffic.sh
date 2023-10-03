#!/usr/bin/env bash
# 
# Script to extract IP source, destination and byte counts from *.pcap.summary.txt files
#
# Got manifest file with lftp
#
BASE_URL="https://stor0.lighthousetb.net"
PATHS_FILES_LABELS=(
  "/dataset/julyreruns/experiment_2/eth1_interface/exp-a-false/,1657930943-c0.chdumb.hdumb.pharos-noenc-a-vtc.false.pcap.summary.txt,vtc1"
  "/dataset/julyreruns/experiment_2/eth1_interface/exp-a-false/,1657931318-c0.chdumb.hdumb.pharos-noenc-a-vtc.false.pcap.summary.txt,vtc2"
  "/dataset/julyreruns/experiment_2/eth1_interface/exp-a-false/,1657931694-c0.chdumb.hdumb.pharos-noenc-a-vtc.false.pcap.summary.txt,vtc3"
  "/dataset/julyreruns/tiered/ipsec/experiment_3/eth1_interface/ptp/exp-a-1080-dash-http1/1659058659-c0.ipsecptp.tiered.pharos-ipsecptp-a-video.1080.dash.http1.pcap.summary.txt
"
)
# ​
# ​
# PATHS_FILES_LABELS=(
#   "/dataset/julyreruns/tiered/wireguard/experiment_3/eth1_interface/sts/exp-b-576-html5-http3/1659059463-c1.wgsts.tiered.pharos-wireguardsts-b-video.576.html5.http3.pcap.summary.txt"
# )
# ​
OUT_FILE="test.csv"
echo "label,dst,src,bytesTo,bytesFrom" >${OUT_FILE} # bytes "to" the destination, bytes "from" the source
for fl in ${PATHS_FILES_LABELS[@]}
do
  P=$(echo $fl | cut -d "," -f 1)
  FILE=$(echo $fl | cut -d "," -f 2)
  LABEL=$(echo $fl | cut -d "," -f 3)
  wget ${BASE_URL}${P}${FILE}
  cat "$FILE" \
    | grep -A 100000 "IPv4 Conversations" \
    | grep -B 100000 -m 1 "=================" \
    | tail -n -4 \
    | head -n -1 \
    | sed -E "s/([0-9]*\.[0-9]*\.[0-9]*\.[0-9]*) *<-> *([0-9]*.[0-9]*.[0-9]*.[0-9]*) *[0-9]* *([0-9]*) *[0-9]* *([0-9]*) *.*/${LABEL},\1,\2,\3,\4/" \
    >> ${OUT_FILE}
done
