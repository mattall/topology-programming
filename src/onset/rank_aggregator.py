data = [
    ("ANS", "ECMP", "/home/mhall/network_stability_sim/ANS_ecmp_minRank.txt", "/home/mhall/network_stability_sim/ANS_ecmp_throughput.txt", "/home/mhall/network_stability_sim/ANS_ecmp_max_congestion.txt"),
    ("ANS", "Ripple*", "/home/mhall/network_stability_sim/ANS_ripple_attack-3x200_minRank.txt", "/home/mhall/network_stability_sim/ANS_ripple_attack-3x200_throughput.txt", "/home/mhall/network_stability_sim/ANS_ripple_attack-3x200_max_congestion.txt"),
    ("CRL", "ECMP", "/home/mhall/network_stability_sim/CRL_ecmp_minRank.txt", "/home/mhall/network_stability_sim/CRL_ecmp_throughput.txt", "/home/mhall/network_stability_sim/CRL_ecmp_max_congestion.txt"),
    ("CRL", "Ripple*", "/home/mhall/network_stability_sim/CRL_ripple_attack-3x200_minRank.txt", "/home/mhall/network_stability_sim/CRL_ripple_attack-3x200_throughput.txt", "/home/mhall/network_stability_sim/CRL_ripple_attack-3x200_max_congestion.txt")]

with open("minRank.csv", "w") as out_fob:
    out_fob.write('Network,Defense,Min Rank,Throughput,Max Congestion\n')
    for d in data:
        with open(d[2],'r') as rank_fob, open(d[3]) as throughput_fob, open(d[4]) as congestion_fob:
            for rank, throughput, congestion in zip(rank_fob.readlines(), throughput_fob.readlines(), congestion_fob.readlines()):
                rank, throughput, congestion = rank.strip(), throughput.strip(), congestion.strip()
                out_fob.write(f"{d[0]},{d[1]},{rank},{throughput},{congestion}\n")