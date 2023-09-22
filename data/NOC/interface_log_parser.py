import re
import json
from sys import argv

testString="""    TenGigE0/1/0/0 is up, line protocol is up
      Interface state transitions: 7
      Hardware is TenGigE, address is 0026.982f.6fc0 (bia 0026.982f.6fc0)
      Layer 1 Transport Mode is LAN
      Description: LAB1
      Internet address is 172.25.26.1/31
      MTU 9172 bytes, BW 10000000 Kbit (Max: 10000000 Kbit)
         reliability 255/255, txload 62/255, rxload 192/255

    HundredGigE0/1/0/1 is up, line protocol is up
      Interface state transitions: 1
      Hardware is TenGigE, address is 0026.982f.6fc1 (bia 0026.982f.6fc1)
      Layer 1 Transport Mode is LAN
      Description: LAB2
      Internet address is 172.25.25.1/31
      MTU 9192 bytes, BW 10000000 Kbit (Max: 10000000 Kbit)
         reliability 255/255, txload 150/255, rxload 20/255

Physical interface: ae0, Enabled, Physical link is Up
  Interface index: 128, SNMP ifIndex: 839
  Description: PEER GOOGLE PNI peering #1 | SOX-56MAR-56MAR-LAG-01852
  Link-level type: Ethernet, MTU: 1514, Speed: 20Gbps, BPDU Error: None,
  MAC-REWRITE Error: None, Loopback: Disabled, Source filtering: Disabled,
  Flow control: Disabled
  Pad to minimum frame size: Disabled
  Minimum links needed: 1, Minimum bandwidth needed: 1bps
  Device flags   : Present Running
  Interface flags: SNMP-Traps Internal: 0x4000
  Current address: 78:19:f7:b8:77:c0, Hardware address: 78:19:f7:b8:77:c0
  Last flapped   : 2023-06-01 04:28:03 EDT (2w0d 09:51 ago)
  Input rate     : 3722794144 bps (448571 pps)
  Output rate    : 432 bps (0 pps)

Physical interface: xe-1/2/3, Enabled, Physical link is Up
  Interface index: 340, SNMP ifIndex: 1419
  Description: TO PEER FACE 10GE via FDP2 17/18 LAG AE11 [PRIV] | SOX-56MAR-56MAR-10GE-02004
  Link-level type: Ethernet, MTU: 1514, MRU: 1522, LAN-PHY mode, Speed: 10Gbps,
  BPDU Error: None, Loop Detect PDU Error: None, MAC-REWRITE Error: None,
  Loopback: None, Source filtering: Disabled, Flow control: Disabled,
  Speed Configuration: Auto
  Pad to minimum frame size: Disabled
  Device flags   : Present Running
  Interface flags: SNMP-Traps Internal: 0x4000
  Link flags     : None
  CoS queues     : 8 supported, 8 maximum usable queues
  Schedulers     : 0
  Current address: 78:19:f7:b8:77:cb, Hardware address: 78:19:f7:b8:70:fa
  Last flapped   : 2023-02-22 13:24:55 EST (16w0d 22:18 ago)
  Input rate     : 3322110800 bps (341882 pps)
  Output rate    : 76960504 bps (54737 pps)
  Active alarms  : None
  Active defects : None
  PCS statistics                      Seconds
    Bit errors                             4
    Errored blocks                         4
  Link Degrade :                      
    Link Monitoring                   :  Disable

Physical interface: xdesd5-13/0, Enabled, Physical link is Up


  Logical interface ae0.0 (Index 375) (SNMP ifIndex 1370)
    Description: [TRCPS] L3 PEER GOOGLE PNI #1 peering | SOX-56MAR-56MAR-VLAN-01853
    Flags: Up SNMP-Traps 0x4004000 Encapsulation: ENET2
    Statistics        Packets        pps         Bytes          bps
    Bundle:
        Input : 8866433990657     448571 9030522376362868   3722794144
        Output: 11854012795780          0 7642467169338500            0
    Adaptive Statistics:
        Adaptive Adjusts:          0
        Adaptive Scans  :          0
        Adaptive Updates:          0
    Protocol inet, MTU: 1500
    Max nh cache: 75000, New hold nh limit: 75000, Curr nh cnt: 1,
    Curr new hold cnt: 0, NH drop cnt: 0
      Flags: Sendbcast-pkt-to-re, Is-Primary
      Addresses, Flags: Is-Default Is-Preferred Is-Primary
        Destination: 74.125.48.32/30, Local: 74.125.48.34,
        Broadcast: 74.125.48.35
    Protocol inet6, MTU: 1500
    Max nh cache: 75000, New hold nh limit: 75000, Curr nh cnt: 1,
    Curr new hold cnt: 0, NH drop cnt: 0
      Flags: Is-Primary
      Addresses, Flags: Is-Default Is-Preferred Is-Primary
        Destination: 2001:4860:1:1::/64, Local: 2001:4860:1:1:0:28fa:0:2
      Addresses, Flags: Is-Preferred
        Destination: fe80::/64, Local: fe80::7a19:f7ff:feb8:77c0
    Protocol multiservice, MTU: Unlimited
"""

#print(testString)
#matchobj=re.search(r"(?P<intrfname>[A-Za-z]+[01]/[01]/[01]/[01]).*?\(bia (?P<bia>[a-f0-9]{4}\.[a-f0-9]{4}\.[a-f0-9]{4})\).*?Internet address is (?P<ipaddr>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d\d{1,5}).*?MTU (?P<mtu>[0-9]+ )",testString,re.M|re.DOTALL)
#if matchobj:
        #print( "\n\nHERE IT IS :: \nInterface : " + matchobj.group('intrfname')  + "\n\nBIA : " + matchobj.group('bia')+ "\n\nIP Address : " +  matchobj.group('ipaddr') + "\n\nMTU : " + matchobj.group('mtu') + "\n\n"+ matchobj.group())

if __name__ == "__main__":
    input_file = argv[1]
    # input_file = "/home/m/src/topology-programming/data/NOC/sox.edu.iu.grnoc.routerproxy/CODAC_Atlanta_GA/showint.log"
    output_file = input_file.split(".")
    output_file = ".".join(output_file[:-1]) + '-update.json'

    with open(input_file, 'r') as fob:
        file_string = fob.read()

    interfaces_1=re.findall(r"Physical interface: (?P<fname>[a-zA-Z]+\d*-?[0-9/]*),.*?link is Up.*?Description: (?P<Description>.*?)\n.*?Speed: (?P<speed>[0-9]+[TGMk]?bps),.*?Hardware Address: (?P<hwaddr>.*?)\n.*?Input rate[ ]+: (?P<inputrate>[0-9]+ bps).*?Output rate[ ]+: (?P<outputrate>[0-9]+ bps)",file_string,re.I|re.M|re.DOTALL)

    interfaces_2=re.findall(r"(?:(?P<EChannel>Ethernet[\d/]+)|(?P<port_chann>Port\-Channel[\d/]+)).*?is up.*?address is (?P<pcChan>[a-f0-9.:]*).*?Description: (?P<Description>.*?\n).*?BW (?P<speed>\d+ [a-zA-Z]+).*?input rate ?(?P<inputrate>[0-9.]+ [a-zA-Z]+){1}.*?output rate ?(?P<outputrate>[0-9.]+ [a-zA-Z]+){1}",file_string,re.I|re.M|re.DOTALL)

    result={}
    for interface in interfaces_1:
            interfaceName=interface[0]
            this_interface = result[interfaceName] = {}
            this_interface["description"] = interface[1]
            this_interface["speed"] = interface[2]
            this_interface["hw address"] = interface[3]
            this_interface["input rate"] = interface[4]
            this_interface["output rate"] = interface[5]
    
    for interface in interfaces_2:
            print(interface)
            
            interfaceName=interface[0] if interface[0] else interface[1]
            this_interface = result[interfaceName] = {}
            this_interface["hw address"] = interface[2]
            this_interface["description"] = interface[3]
            this_interface["speed"] = interface[4]
            this_interface["input rate"] = interface[5]
            this_interface["output rate"] = interface[6]

    with open(output_file, 'w') as fob:
         json.dump(result, fob, indent=4)
