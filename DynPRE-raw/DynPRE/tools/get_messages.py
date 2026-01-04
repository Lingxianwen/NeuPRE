from netzob.all import PCAPImporter
import socket
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


messages = PCAPImporter.readFile(sys.argv[1]).values()


received = ""
for i, message in enumerate(messages):
    print(
        message.__str__().split("'")[0],
        bytes(message.__str__().split("'")[1], encoding="utf-8").hex(),
    )
