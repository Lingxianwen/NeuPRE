# -*- coding: utf-8 -*
import logging
import os
import socket

from Constants import *
from MessageRewriter import MessageRewriter
from RewriteRule import Location, RewriteRule


def compare_bytes(bytes1, bytes2):
    result = []
    in_diff = False
    diff_start = None

    all_diff_range = []

    for i, (a, b) in enumerate(zip(bytes1, bytes2)):
        if a == b:
            result.append("{:02x}".format(a))
            if in_diff:
                logging.debug(f"Difference range: {diff_start}-{i-1}")
                all_diff_range.append((diff_start, i - 1))
                in_diff = False
                diff_start = None
        else:
            result.append("__")
            if not in_diff:
                in_diff = True
                diff_start = i

    if in_diff:
        logging.debug(f"Difference range: {diff_start}-{i}")
        all_diff_range.append((diff_start, i))

    logging.debug("".join(result))

    return all_diff_range


def find_subbytes_occurrences(main_bytes, sub_bytes):
    occurrences = []
    start = 0
    while True:
        index = main_bytes.find(sub_bytes, start)
        if index != -1:
            occurrences.append((index, index + len(sub_bytes) - 1))
            start = index + 1
        else:
            break

    return occurrences


class Sender:
    def __init__(self) -> None:
        self.SocketSender = None
        self.rewriter = MessageRewriter()

    def CreateConnection(self, ip, port):  # TCP
        logging.debug(f"Connecting to {ip}:{port}")
        self.SocketSender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.SocketSender.settimeout(3)
        try:
            self.SocketSender.connect((ip, port))
        except Exception as e:
            logging.error(f"Connection to {ip}:{port} failed: {e}")
            return False
        # self.SocketSender.recv(1024)
        return True

    def DisConnection(self):
        logging.debug("Disconnecting")
        self.SocketSender.close()

    def SessIDDetector(self, trace, rewrite_rules=[]):
        logging.debug(f"SessIDDetector started")
        self.rewriter.set_rules(rewrite_rules)
        self.CreateConnection(
            ip=trace.M[0].raw["IP"].strip(), port=int(trace.M[0].raw["Port"])
        )
        for index, message in enumerate(trace.M):
            logging.debug(f"DryRun sending message {index}")
            response = self.sendMessage(message, index)
            if len(response) != len(message.raw["Response"]):
                return False, None
            if response != message.raw["Response"]:
                if self.rewriter.has_rule(index):
                    continue
                diff_ranges = compare_bytes(response, message.raw["Response"])
                added_rule = []

                for diff_range in diff_ranges:
                    is_referenced = False
                    for rule in rewrite_rules:
                        for reference in rule.references:
                            if (
                                reference.message_index == index
                                and reference.direction == DIRECTION.RESPONSE
                                and (
                                    reference.byte_region[0] <= diff_range[0]
                                    and diff_range[1] <= reference.byte_region[1]
                                )
                            ):
                                logging.debug(
                                    f"Rule {rule} already added for message {index} range {diff_range}, skip"
                                )
                                is_referenced = True
                                break
                        if is_referenced:
                            break
                    if is_referenced:
                        continue

                    diff_content = message.raw["Response"][
                        diff_range[0] : diff_range[1] + 1
                    ]
                    source = Location(index, DIRECTION.RESPONSE, diff_range)
                    references = []
                    for _ in range(index + 1, len(trace.M)):
                        request = trace.M[_].raw["Content"]
                        response = trace.M[_].raw["Response"]
                        found_regions = find_subbytes_occurrences(request, diff_content)
                        if len(found_regions) > 0:
                            logging.debug(
                                f"Diff content {diff_content.hex()} found in Req@{_}: {found_regions}"
                            )
                            for region in found_regions:
                                references.append(
                                    Location(_, DIRECTION.REQUEST, region)
                                )
                        # response
                        found_regions = find_subbytes_occurrences(
                            response, diff_content
                        )
                        if len(found_regions) > 0:
                            logging.debug(
                                f"Diff content {diff_content.hex()} found in Rsp@{_}: {found_regions}"
                            )
                            for region in found_regions:
                                references.append(
                                    Location(_, DIRECTION.RESPONSE, region)
                                )
                    if len(references) > 0:
                        rule = RewriteRule(source, references, 0)
                        added_rule.append(rule)
                        logging.debug(f"Going to added rule: {rule}")
                if len(added_rule) > 0:
                    self.DisConnection()
                    logging.debug("Trying again with new rules")
                    is_ok, final_rewrite_rules = self.SessIDDetector(
                        trace, rewrite_rules=rewrite_rules + added_rule
                    )
                    if is_ok:
                        return True, final_rewrite_rules
                    else:
                        return False, None

        self.DisConnection()
        return True, rewrite_rules

    def DryRunSend(self, trace):  # send a trace of messages (work for DryRun)
        # trace.display()  # Test only
        logging.debug(f"DryRun started.")
        self.rewriter.set_rules(trace.RR)
        self.CreateConnection(
            ip=trace.M[0].raw["IP"].strip(), port=int(trace.M[0].raw["Port"])
        )
        for index, message in enumerate(trace.M):
            logging.debug(f"DryRun sending message {index}")
            response = self.sendMessage(message, index)
            if response == "#error":
                self.DisConnection()
                return True
            if len(response) != len(message.raw["Response"]):
                raise Exception(
                    "The response length does not match the expected length"
                )
            trace.R.append(response)
        self.DisConnection()
        return trace

    def ProbeSend(
        self, trace, index: int
    ) -> (
        bytes
    ):  # send a trace of messages (work for Probe) until index-th message, collect the response of index-th message
        self.rewriter.set_rules(trace.RR)
        self.CreateConnection(
            ip=trace.M[0].raw["IP"].strip(), port=int(trace.M[0].raw["Port"])
        )
        for i in range(index + 1):
            response = self.sendMessage(trace.M[i], i)
            if response == b"#error":
                self.DisConnection()
                return b"#error"
            elif response == b"#crash":
                self.DisConnection()
                return b"#crash"
            if i == index:
                res = response
        self.DisConnection()
        return res

    def MutationSend(self, trace, index):  # send a trace of messages
        self.rewriter.set_rules(trace.RR)
        for i in range(len(trace.M)):
            response = self.sendMessage(trace.M[i])
            if response == "#error":
                return "#error"
            elif response == "#crash":
                return "#crash"
            if i == index:
                res = response

        pool = trace.PR[index]
        scores = trace.PS[index]
        for i in range(len(pool)):
            c = SimilarityScore(pool[i].strip(), res.strip())
            if c >= scores[i]:
                return ""
        return "#interesting-" + str(index)

    def sendMessage(
        self, message, sequence_num: int, time=0
    ) -> bytes:  # send a message
        if "IP" in message.headers and "Port" in message.headers:  # socket
            ip = message.raw["IP"].strip()
            port = int(message.raw["Port"])
            content = self.rewriter.send_rewrite(message.raw["Content"], sequence_num)
            logging.debug(f"Sending message: {content}, len: {len(content)}")

            try:
                self.SocketSender.send(content)
                response = self.SocketSender.recv(2048)
                logging.debug(f"Response {response}")
            except socket.timeout as e:
                logging.debug(f"Response timeout")
                return b"#timeout"
            return self.rewriter.recv_rewrite(response, sequence_num)
        else:
            raise ("Error : IP and Port of target should be included in input files")


def EditDistanceRecursive(str1, str2):
    if str1 is None or str2 is None:
        raise Exception("Input strings cannot be None")
    elif len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(
                edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d
            )
    return edit[len(str1)][len(str2)]


def SimilarityScore(str1, str2):
    if str1 is None or str2 is None:
        raise Exception("Input strings cannot be None")
    elif len(str1) == 0 or len(str2) == 0:
        raise Exception("Input strings cannot be empty")
    ED = EditDistanceRecursive(str1, str2)
    return round((1 - (ED / max(len(str1), len(str2)))) * 100, 2)
