# -*- coding: utf-8 -*
class Trace:
    def __init__(self) -> None:
        self.M = []  # Message List - message type
        self.R = []  # Response List - string
        self.PR = []  # Probe message response pool - 2d list
        self.SG = []  # Field Identification Result
        self.PRA = []  # All responses of probe messages
        self.RR = []  # Learned rewrite rules for this trace
        self.PS = []
        self.PI = []

    def append(self, message):
        self.M.append(message)

    def response(self, response):
        self.R.append(response)

    def display(self):
        for i in range(0, len(self.M)):
            print("Message index: ", i + 1)
            for header in self.M[i].headers:
                print(header, " : ", self.M[i].raw[header])
            print("Response:")
            print(self.R[i])
            if self.PR and self.PS and self.PI:
                print("Probe Result:")
                print("PI")
                print(self.PI[i])
                print("PR and PS")
                for n in range(len(self.PR[i])):
                    print("(" + str(n) + ") " + self.PR[i][n])
                    print(self.PS[i][n])


class Message:
    headers = []  # Header List
    raw = {}  # Header and corresponding content

    def __init__(self) -> None:
        self.headers = []
        self.raw = {}

    def append(self, line) -> None:
        if ":" in line:
            sp = line.split(":")
            if sp[0] in self.headers:
                print("Error. Message headers '", sp[0], "' is duplicated.")
            else:
                self.headers.append(sp[0])
                self.raw[sp[0]] = line[(line.index(":") + 1) :]
