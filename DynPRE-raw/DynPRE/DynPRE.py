# -*- coding: utf-8 -*
import csv
import getopt
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import Constants
from FieldIdentify import FieldIdentifier
from LogManager import init_logger
from netzob.all import Format, RawMessage, Symbol
from Reader import readInputFile, readPcap
from Record import RecordWriter
from Sender import Sender

#  Golbal var
traces = []
outputfold = ""
text_mode = False
read_input_file = False
datasize = -1


def ByteFlip(message: bytes, index_range: Tuple[int, int]) -> bytes:
    res = message[: index_range[0]]
    for _ in range(index_range[0], index_range[1] + 1):
        res += (255 - message[_]).to_bytes(1, byteorder="big")
    return res + message[index_range[1] + 1 :]


def DelByte(message: bytes, index: int) -> bytes:
    return message[:index] + message[index + 1 :]


# read the input fold and store them as traces
def readInputFold(fold):
    global read_input_file
    traces = []
    files = os.listdir(fold)
    if not read_input_file:
        read_func = readPcap
    else:
        read_func = readInputFile
    for file in files:
        print("Loading file: ", os.path.join(fold, file))
        traces.append(read_func(os.path.join(fold, file)))
    return traces


# Try to use the input given for a complete communication.
# The func is used to test whether the input meets the requirements or whether there are other problems
def dryRun(traces) -> Dict:
    m = Sender()
    all_rewrite_rules = {}
    for i in range(0, len(traces)):
        learn_times = 0
        while learn_times < Constants.REWRITE_RULE_MAX_LEARN_TIMES:
            is_ok, final_rewrite_rules = m.SessIDDetector(traces[i])
            if is_ok:
                final_rewrite_rules_str = "\n".join(
                    [str(rule) for rule in final_rewrite_rules]
                )
                logging.debug(
                    f"SessIDDetector finished, final rewrite rules:\n{final_rewrite_rules_str}"
                )
                all_rewrite_rules[i] = [r.to_dict() for r in final_rewrite_rules]
                traces[i].RR = final_rewrite_rules
                break
            learn_times += 1
        trace = m.DryRunSend(traces[i])
        traces[i] = trace
        logging.info(f"Trace {i} dry run finished")
    return all_rewrite_rules


# Calculate the edit distance of two string
def CalEditDistance(str1, str2):
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


# Calculate the similarity score of two string
def CalSimilarity(str1, str2):
    if max(len(str1), len(str2)) == 0:
        return round(100.0, 2)
    ED = CalEditDistance(str1, str2)
    return round((1 - (ED / max(len(str1), len(str2)))) * 100, 2)


def get_diff_without_rand(
    s1, s2, exclude_ranges: List[Tuple]
) -> List:  # Ensure s1 and s2 are of the same length
    no_zero_index = []
    original_diff = [a ^ b for (a, b) in zip(s1, s2)]
    for i, c in enumerate(original_diff):
        if c != 0:
            flag = False
            for r in exclude_ranges:
                if r[0] <= i <= r[1]:
                    flag = True
                    break
            if not flag:
                no_zero_index.append(i)
    return no_zero_index


# get nonzero ranges of a list
def get_nonzero_range(a: List) -> List[Tuple]:
    # Compute list "b" by replacing any non-zero value of "a" with 1
    b = list(map(int, [i != 0 for i in a]))

    # Compute ranges of 0 and ranges of 1, and only record ranges of 1
    idx = []  # result list of tuples
    start = 0  # index of first element of each range of zeros or non-zeros
    for n in range(len(b)):
        if (n + 1 == len(b)) or (b[n] != b[n + 1]):
            # Here: EITHER it is the last value of the list
            #       OR a new range starts at index n+1
            if b[start]:
                idx.append((start, n))
            start = n + 1

    return idx


# merge two rand mask
def merge_rand_mask(l1: List[Tuple], l2: List[tuple]) -> List[Tuple]:
    res = []
    mx = 0
    for a in l1 + l2:
        mx = max(mx, a[1])
    pos = [0 for _ in range(mx + 1)]
    for a in l1 + l2:
        for i in range(a[0], a[1] + 1):
            pos[i] = 1
    return get_nonzero_range(pos)


# check if l1 is compatible to l2
def rand_mask_unidirection_compatibility(l1: List[Tuple], l2: List[Tuple]) -> bool:
    for a in l1:
        flag = False
        for b in l2:
            if not (a[1] < b[0] or a[0] > b[1]):
                flag = True
                break
        if not flag:
            return False
    return True


# check if two rand mask are compatible
def rand_mask_compatibility(l1: List[Tuple], l2: List[Tuple]) -> bool:
    return rand_mask_unidirection_compatibility(
        l1, l2
    ) and rand_mask_unidirection_compatibility(l2, l1)


def EliminateOthers(message, start, end):
    result = b""
    for i in range(len(message)):
        if i < start or i >= end:
            result += (255 - message[i]).to_bytes(1, byteorder="big")
        else:
            result += message[i].to_bytes(1, byteorder="big")
    return result


def Analyze(Trace):

    print("*** Analyze ")
    m = Sender()
    offset = []
    for index in range(len(Trace.M)):

        logging.debug(f"Analyze message@{index}")
        responsePool = []

        print(Trace.M[index].raw["Content"])  # test only
        # original message
        response1 = m.ProbeSend(Trace, index)  # send the probe

        responsePool.append(response1)

        # probe process
        for i in range(0, len(offset) - 1):
            temp = Trace.M[index].raw["Content"]
            Trace.M[index].raw["Content"] = EliminateOthers(
                Trace.M[index].raw["Content"], offset[i], offset[i + 1]
            )

            response1 = m.ProbeSend(Trace, index)  # send the probe message   #######
            responsePool.append(response1)

            Trace.M[index].raw["Content"] = temp  # restore the message

        print("Response Pull:\n")
        for response in responsePool:
            print(response)
    sys.exit(0)


# Probe byte@{byte_index} of message@{pkt_index} and return the response
def ProbeOneStep(Trace, pkt_index: int, index_range: Tuple[int, int]) -> bytes:
    m = Sender()
    temp = Trace.M[pkt_index].raw["Content"]
    Trace.M[pkt_index].raw["Content"] = ByteFlip(
        Trace.M[pkt_index].raw["Content"], index_range
    )
    response = m.ProbeSend(Trace, pkt_index)
    Trace.M[pkt_index].raw["Content"] = temp  # restore the message
    return response


def get_rand_mask(
    s1: bytes, s2: bytes
) -> List[Tuple]:  # ensure s1 and s2 are of the same length
    assert len(s1) == len(s2)
    return get_nonzero_range([a ^ b for (a, b) in zip(s1, s2)])


# Use heuristics to detect the meaning of each byte in the message
def Probe(Trace, probe_size=None, record_writer: Optional[RecordWriter] = None):

    print("*** Probe ")
    m = Sender()
    cnt = 0
    for index in range(len(Trace.M)):
        logging.info(f"Probing message@{index}")
        responsePool = []
        responsePool_all = [Trace.M[index].raw["Response"]]
        similarityScore = []
        probeResponseIndex = []

        # original message
        response1 = m.ProbeSend(Trace, index)  # send the probe message
        time.sleep(Constants.SESSION_INTERVAL_SHORT)
        logging.debug("Sending twice")
        response2 = m.ProbeSend(Trace, index)  # send the probe message twice

        responsePool.append(
            response1
        )  # responsePool stores the response to a single message
        responsePool_all.append(response1)
        assert len(response1) == len(responsePool_all[0])
        similarityScore.append(CalSimilarity(response1.strip(), response2.strip()))

        start_index_response = b""
        start_index_response2 = b""
        start_index = -1
        cumulative_rand_mask = []
        segmentation = []

        # probe process
        for i in range(0, len(Trace.M[index].raw["Content"])):
            logging.info(
                "Probe (message, index) = (" + str(index) + ", " + str(i) + ")\n"
            )

            response1 = ProbeOneStep(Trace, index, (i, i))
            time.sleep(Constants.SESSION_INTERVAL_SHORT)
            response2 = ProbeOneStep(
                Trace, index, (i, i)
            )  # send the probe message twice
            if len(response1) != len(response2):
                time.sleep(Constants.SESSION_INTERVAL_LONG)
                response3 = ProbeOneStep(Trace, index, (i, i))
                if len(response3) == len(response1):
                    response2 = response3
                elif len(response3) == len(response2):
                    response1 = response2
                    response2 = response3
                else:
                    raise (
                        "len(response3)!=len(response1) and len(response3)!=len(response2)"
                    )

            responsePool_all.append(response1)
            # responsePool_all.append(response2)

            if responsePool:
                flag = True
                for j in range(0, len(responsePool)):
                    target = responsePool[j]
                    score = similarityScore[j]
                    c = CalSimilarity(target.strip(), response1.strip())
                    if c >= score:
                        flag = False
                        probeResponseIndex.append(j)
                        break
                if flag:
                    responsePool.append(response1)
                    similarityScore.append(
                        CalSimilarity(response1.strip(), response2.strip())
                    )
                    probeResponseIndex.append(j + 1)

            assert len(response1) == len(response2)

            start_index_response2 = b""
            if i > 0:
                if response1 != start_index_response:
                    if len(response1) != len(start_index_response):
                        segmentation.append(i)
                        start_index_response = response1
                        start_index = i
                        cumulative_rand_mask = get_rand_mask(response1, response2)
                    else:
                        time.sleep(Constants.SESSION_INTERVAL_SHORT)
                        start_index_response2 = ProbeOneStep(
                            Trace, index, (start_index, start_index)
                        )
                        if len(start_index_response) != len(start_index_response2):
                            time.sleep(Constants.SESSION_INTERVAL_LONG)
                            start_index_response2 = ProbeOneStep(
                                Trace, index, (start_index, start_index)
                            )
                            if len(start_index_response) != len(start_index_response2):
                                raise (
                                    "len(start_index_response)!=len(start_index_response2)"
                                )
                        cumulative_rand_mask = merge_rand_mask(
                            cumulative_rand_mask,
                            get_rand_mask(start_index_response, start_index_response2),
                        )
                        cur_rand_mask = get_rand_mask(response1, response2)
                        if not rand_mask_compatibility(
                            cumulative_rand_mask, cur_rand_mask
                        ):
                            segmentation.append(i)
                            start_index_response = response1
                            start_index = i
                            cumulative_rand_mask = get_rand_mask(response1, response2)
                        else:
                            cumulative_rand_mask = merge_rand_mask(
                                cumulative_rand_mask, cur_rand_mask
                            )
                            response_diff = get_diff_without_rand(
                                response2, start_index_response2, cumulative_rand_mask
                            )
                            if (len(response_diff) == 0) or (
                                len(response_diff) == 2
                                and (
                                    response_diff[1] - response_diff[0]
                                    == i - start_index
                                )
                            ):
                                pass
                            else:
                                segmentation.append(i)
                                start_index_response = response1
                                start_index = i
                                cumulative_rand_mask = get_rand_mask(
                                    response1, response2
                                )
            else:
                segmentation.append(i)
                start_index_response = response1
                start_index = i
                cumulative_rand_mask = get_rand_mask(response1, response2)

        segmentation.append(len(Trace.M[index].raw["Content"]))
        Trace.PR.append(responsePool)
        Trace.PS.append(similarityScore)
        Trace.PI.append(probeResponseIndex)
        Trace.SG.append(segmentation)
        Trace.PRA.append(responsePool_all)

        segment_mutation_responses = []

        for seg_index in range(len(segmentation) - 1):
            if segmentation[seg_index] + 1 == segmentation[seg_index + 1]:
                continue
            segment_mutation_responses.append(
                ProbeOneStep(
                    Trace,
                    index,
                    (segmentation[seg_index], segmentation[seg_index + 1] - 1),
                )
            )

        responses_to_alignment = {}

        for _m in responsePool_all + segment_mutation_responses:
            responses_to_alignment[_m] = 1

        responses_to_alignment = responses_to_alignment.keys()

        if record_writer is not None:
            record_writer.write_record(Trace, index)

            record_writer.write_responses_segmentation(
                splitResponse(
                    [responsePool_all[0]],
                    [
                        r
                        for r in responses_to_alignment
                        if r != b"#timeout" and len(r) == len(responsePool_all[0])
                    ],
                ),
                Constants.ResponseSegmentationFile,
            )
        cnt += 1
        if (probe_size is not None) and (cnt >= probe_size):
            break

    return Trace, cnt


def splitResponse(
    analyzed_messages: List, responsePool: List
) -> List[Tuple[bytes, List]]:  # list of tuple (hexstream, split_index_list)
    for analyzed_message in analyzed_messages:
        if analyzed_message not in responsePool:
            raise ("analyzed_message not in responsePool")

    logging.debug(f"Response count: {len(responsePool)}")
    combMessage = []
    for message in responsePool:
        combMessage.append(RawMessage(message))
    symbol = Symbol(messages=combMessage)
    Format.splitAligned(symbol, doInternalSlick=False)

    # print(symbol)

    analMessage = []
    for message in analyzed_messages:
        analMessage.append(RawMessage(message))
    symbol.messages = analMessage

    messageCells = symbol.getMessageCells()
    msg_to_index = {}
    for message in symbol.messages:
        cur = 0
        index = [0]
        for field in messageCells[message]:
            flen = len(field.hex()) // 2
            if flen:
                cur = cur + len(field.hex()) // 2
                index.append(cur)
        msg_to_index[hash(message.data.hex())] = index

    return [(m.hex(), msg_to_index[hash(m.hex())]) for m in analyzed_messages]


def clusterSplitResponse(
    analyzed_messages: List, responsePool: List
) -> List[Tuple[bytes, List]]:  # list of tuple (hexstream, split_index_list)
    for analyzed_message in analyzed_messages:
        if analyzed_message not in responsePool:
            raise ("analyzed_message not in responsePool")

    combMessage = []
    for message in responsePool:
        combMessage.append(RawMessage(message))
    symbols = Format.clusterByAlignment(messages=combMessage, internalSlick=True)

    print(f"count: {len(responsePool)}, symbol num: {len(symbols)}")
    msg_to_index = {}
    for symbol in symbols:
        print(symbol._str_debug())
        symbol_analMessage = []
        for x in analyzed_messages:
            for y in symbol.messages:
                if x == y.data:
                    symbol_analMessage.append(y)
                    break
        if len(symbol_analMessage) == 0:
            continue
        symbol.messages = symbol_analMessage  # Replace symbol's message, so that only the message in analyzed_messages needs to be parsed

        messageCells = symbol.getMessageCells()

        for message in symbol.messages:
            cur = 0
            index = [0]
            for field in messageCells[message]:
                flen = len(field.hex()) // 2
                if flen:
                    cur = cur + len(field.hex()) // 2
                    index.append(cur)
            msg_to_index[hash(message.data.hex())] = index

    return [(m.hex(), msg_to_index[hash(m.hex())]) for m in analyzed_messages]


def mySplitResponse(
    analyzed_messages: List, responsePool: List
) -> List[Tuple[bytes, List]]:  # ensure all the messages are of the same length
    for analyzed_message in analyzed_messages:
        if analyzed_message not in responsePool:
            raise ("analyzed_message not in responsePool")
    bytes_values = []
    for i in range(len(analyzed_messages[0])):
        byte_values = {}
        for message in responsePool:
            byte_values[message[i]] = 1
        if len(byte_values) > 1:
            bytes_values.append(1)  # dynamic
        else:
            bytes_values.append(0)  # static
    seg_index = [0]
    for i in range(len(analyzed_messages[0]) - 1):
        if bytes_values[i] != bytes_values[i + 1]:
            seg_index.append(i + 1)
    seg_index.append(len(analyzed_messages[0]))
    return [(m.hex(), seg_index) for m in analyzed_messages]


def getArgs(argv):
    inputfold = ""
    outputfold = ""
    datasize = None
    text_mode = False
    read_input_file = False

    try:
        opts, args = getopt.getopt(argv, "htfi:o:s:", ["ifold=", "ofold=", "size="])
    except getopt.GetoptError:
        print(
            f"DynPRE.py -i <inputfold> -o <outputfold> [-t/--textual] [-f/--file] [-s/--size]"
        )
        sys.exit(2)
    # IPython.embed()
    for opt, arg in opts:
        if opt == "-h":
            print(
                f"DynPRE.py -i <inputfold> -o <outputfold> [-t/--textual] [-f/--file] [-s/--size]"
            )
            sys.exit()
        elif opt in ("-i", "--ifold"):
            inputfold = arg
        elif opt in ("-o", "--ofold"):
            outputfold = arg
        elif opt in ("-s", "--size"):
            datasize = int(arg)
        elif opt in ("-t", "--textual"):
            text_mode = True
        elif opt in ("-f", "--file"):
            read_input_file = True

    print("Input fold: ", inputfold)
    print("Output fold: ", outputfold)
    print("Textual mode: ", text_mode)
    print("Size: ", datasize)
    print("Use file as input: ", read_input_file)

    if datasize != None:
        datasize = math.ceil(datasize / 2)

    return inputfold, outputfold, text_mode, read_input_file, datasize


def main(argv):

    init_logger(level=logging.INFO)

    global traces, outputfold, text_mode, read_input_file, datasize

    inputfold, outputfold, text_mode, read_input_file, datasize = getArgs(argv)

    traces = readInputFold(inputfold)

    logging.info("Dry running...")
    record_writer = RecordWriter(outputfold)
    all_rewrite_rules = dryRun(traces)
    record_writer.write_rewrite_rules(json.dumps(all_rewrite_rules))
    remaining_size = datasize
    for i in range(len(traces)):
        logging.debug(f"Probe file@{i}")
        record_writer.new_seed(i)
        traces[i], probe_size = Probe(traces[i], remaining_size, record_writer)
        if remaining_size != None:
            remaining_size -= probe_size
            if remaining_size <= 0:
                break
    logging.info("Analyzing segmentation and refining")
    field_identifier = FieldIdentifier(out_dir=outputfold)
    field_identifier.run()
    record_writer.write_type_field(field_identifier.type_fields)
    record_writer.write_varified_field(field_identifier.varified_fields)


if __name__ == "__main__":
    main(sys.argv[1:])
