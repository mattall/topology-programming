f1 = "scripts/TDSC/legacy/net-ANS_TE-mcf_-trafficDscription-coremelt_repeat-False_failure-False_trafficFile-ANS_coremelt_every_link_2_fallowTXP-1_opticalStrategy-onset_FTXAllocPolicy-file"
f2 = "scripts/TDSC/args/TXP_link_rank/ans/mcf.txt"

# thesis: these should have **Some** overlap concerning f1 and f2

fob1 = open(f1, 'r')
fob2 = open(f2, 'r')

lines_1 = list(fob1.readlines())
lines_2 = list(fob2.readlines())

common_lines = lines_1.intersection(lines_2)

print(f"common lines: {len(common_lines)}")
A = lines_1.pop().split()
B = lines_2.pop().split()
for a,b in zip(A,B):
  print(f"{a}\t\t\t, {b}")