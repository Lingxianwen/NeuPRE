# -*- coding: utf-8 -*
import logging

from netzob.all import PCAPImporter
from Trace import Message, Trace


def readPcap(file):
    trace = Trace()
    pcap = PCAPImporter.readFile(file).values()
    server_ip = pcap[0].destination.split(":")[0]
    server_port = pcap[0].destination.split(":")[1]
    last_request = None
    for cnt, m in enumerate(pcap):
        message = Message()
        if m.destination != pcap[0].destination:
            continue
        message.append("IP:" + server_ip)
        message.append("Port:" + server_port)
        message.append("Content:")
        message.raw["Content"] = m.data
        message.append("Response:")
        if cnt + 1 < len(pcap) and pcap[cnt + 1].source == pcap[0].destination:
            message.raw["Response"] = pcap[cnt + 1].data
        else:
            logging.warn(f"No response for message@{cnt}")
        trace.append(message)

    return trace


# read the input file and store it as trace
def readInputFile(file):
    global text_mode
    s = Trace()
    lines = []
    with open(file, "r") as f:
        lines = f.read().split("\n")
    for i in range(0, len(lines)):
        # print(lines[i])
        if "========" in lines[i]:
            mes = Message()
            for j in range(i + 1, len(lines)):
                if "========" in lines[j]:
                    i = j
                    break
                if ":" in lines[j]:
                    mes.append(lines[j])
            if not text_mode:
                mes.raw["Content"] = bytes.fromhex(mes.raw["Content"].strip())
            else:
                mes.raw["Content"] = (mes.raw["Content"].strip() + "\r\n").encode()
            s.append(mes)
    return s
