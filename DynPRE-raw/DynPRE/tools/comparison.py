import pyshark
from typing import Dict, List
import sys
import csv
import logging
import os
import ast

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_score_new(ground_truth_lst, lst, ground_truth_use_byte=True):
    if not ground_truth_use_byte:
        new_lst = [i * 2 for i in ground_truth_lst]
        ground_truth_lst = new_lst
    pkt_len = ground_truth_lst[-1]
    ground_truth_lst = ground_truth_lst[1:-1]
    lst = lst[1:-1]

    positives = len(ground_truth_lst)
    # negatives = (pkt_len - 1) - positives
    tp = 0  # infered_true_boundaries
    fp = 0  # infered_false_boundaries
    fn = 0  #
    tn = 0  #

    for i in lst:
        if i in ground_truth_lst:
            tp += 1
        else:
            fp += 1
    for i in ground_truth_lst:
        if i not in lst:
            fn += 1
    tn = (pkt_len - 1) - (tp + fp + fn)

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    if fp + tn == 0:
        fpr = 0.0
    else:
        fpr = fp / (fp + tn)

    accuracy = (tp + tn) / (pkt_len - 1)

    return [precision, recall, accuracy, f1, fpr]


def get_score(ground_truth_lst, lst, ground_truth_use_byte=True):
    if not ground_truth_use_byte:
        new_lst = [i * 2 for i in ground_truth_lst]
        ground_truth_lst = new_lst

    if ground_truth_lst[-1] != lst[-1]:
        logging.warn(
            "Find one split result whose end index doesn't match"
        )  # Netplier's dhcp and tftp

    pos = [0 for _ in range(ground_truth_lst[-1] + 1)]

    for index in ground_truth_lst:
        pos[index] = 1

    cover_num = 0
    in_num = 0
    accurate_num = 0
    for i in range(len(lst) - 1):
        if (pos[lst[i]] == 1) and (pos[lst[i + 1]] == 1):
            cover_num += 1
            if sum(pos[lst[i] + 1 : lst[i + 1]]) == 0:
                accurate_num += 1
                logging.debug(f"accurate: {lst[i]}, {lst[i+1]}")
            else:
                logging.debug(f"cover field: {lst[i]}, {lst[i+1]}")
        elif sum(pos[lst[i] + 1 : lst[i + 1]]) == 0:
            in_num += 1
            logging.debug(f"in field: {lst[i]}, {lst[i+1]}")
        else:
            logging.debug(f"error field: {lst[i]}, {lst[i+1]}")
    return [
        (cover_num + in_num) / (len(lst) - 1),
        accurate_num / (len(ground_truth_lst) - 1),
    ]


def dfs(dct: Dict, index: List = []):
    for k, v in dct.items():
        if k.endswith("_raw") and isinstance(v, list):
            fetch(v, index=index)
        elif k.endswith("_tree") or isinstance(v, dict):
            if isinstance(v, dict):
                dfs(v, index=index)
            elif isinstance(v, list):
                for vv in v:
                    dfs(vv, index=index)


def fetch(lst, index=[]):
    if isinstance(lst[0], list):
        for m in lst:
            fetch(m, index=index)
    else:
        if lst[0] != "field length invalid!":
            index.append(int(lst[1]))
            index.append(int(lst[1]) + int(lst[2]))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"python {sys.argv[0]} input_file_dir filter layer_num out_dir")
        sys.exit(0)

    ground_truth_file = os.path.join(
        sys.argv[4], os.path.split(sys.argv[1])[1] + ".ground_truth.csv"
    )
    csvfile = open(ground_truth_file, "w")
    writer = csv.writer(csvfile)
    writer.writerow(["Hexstream", "Split Indexes", "Splited Hexstream"])

    files = os.listdir(sys.argv[1])
    files.sort()

    ground_truth = {}
    for file in files:
        if file.endswith(".pcap"):
            logging.info(f"Analyzing groud truth of file {file}")
            # import pcap
            cap = pyshark.FileCapture(
                os.path.join(sys.argv[1], file),
                display_filter=sys.argv[2],
                include_raw=True,
                use_json=True,
            )

            _ = 0
            for packet in cap:

                ans = []
                layers_dict = {}
                _ += 1
                logging.debug(f"Processing packet@{_}")
                for layer in range(int(sys.argv[3]), len(packet.layers)):
                    layers_dict = dict(
                        **layers_dict, **packet.layers[layer]._all_fields
                    )
                dfs(layers_dict, ans)  # packet.xxx._all_fields

                res = list(set(ans))
                res.sort()

                base = res[0]
                for i in range(len(res)):
                    res[i] -= base

                pkt_layer = packet.get_raw_packet().hex()[base * 2 :]
                if res[-1] != len(pkt_layer) // 2:
                    if res[-1] > len(pkt_layer) // 2:
                        raise Exception("Field subscript exceeds packet length!")
                    else:
                        res.append(len(pkt_layer) // 2)

                pkt_layer_split = " ".join(
                    pkt_layer[2 * res[i] : 2 * res[i + 1]] for i in range(len(res) - 1)
                )

                writer.writerow([pkt_layer, res, pkt_layer_split])
                if pkt_layer not in ground_truth:
                    ground_truth[pkt_layer] = (res, pkt_layer_split)
                else:
                    if ground_truth[pkt_layer] != (res, pkt_layer_split):
                        logging.warn(
                            f"Find a packet that has more than one ground truth"
                        )

    files = os.listdir(sys.argv[4])
    files.sort()
    request_average_scores = []
    request_refine_average_scores = []
    response_average_scores = []
    response_refine_average_scores = []
    request_cnt = 0
    response_cnt = 0
    for file in files:
        if (
            file.endswith(".csv")
            and not file.endswith(".perf.csv")
            and not file.endswith(".ground_truth.csv")
        ):
            logging.info(f"Comparing {file} with ground truth")
            compared_csvfile = open(os.path.join(sys.argv[4], file), "r")
            compared_reader = csv.reader(compared_csvfile)
            header = next(compared_reader)

            output_csvfile = open(os.path.join(sys.argv[4], file + ".perf.csv"), "w")

            writer = csv.writer(output_csvfile)
            writer.writerow(
                [
                    "Packet No.",
                    "Hexstream",
                    "Split Indexes",
                    "Splited Hexstream",
                    "True Splited Indexes",
                    "True Splited Hexstream",
                    "Correctness (Cover or In Field Num. / Inferred Field Num.)",
                    "Perfection (Accurate Field Num. / Correct Field Num.)",
                    "Precision",
                    "Recall",
                    "Accuracy",
                    "F1",
                    "FPR",
                ]
            )

            _cnt = 0
            scores = [0, 0]
            scores_new = [0, 0, 0, 0, 0]
            for index, compared_line in enumerate(compared_reader):
                if compared_line[0] not in ground_truth:
                    logging.warn(
                        f"In {sys.argv[2]}, find hexstream@{index} that don't have ground truth in {sys.argv[1]}"
                    )
                    continue
                _cnt += 1
                score = get_score(
                    ground_truth[compared_line[0]][0],
                    ast.literal_eval(compared_line[1]),
                    ground_truth_use_byte=False if len(sys.argv) == 4 else True,
                )
                score_new = get_score_new(
                    ground_truth[compared_line[0]][0],
                    ast.literal_eval(compared_line[1]),
                    ground_truth_use_byte=False if len(sys.argv) == 4 else True,
                )
                writer.writerow(
                    [
                        index + 1,
                        compared_line[0],
                        ast.literal_eval(compared_line[1]),
                        compared_line[2],
                        ground_truth[compared_line[0]][0],
                        ground_truth[compared_line[0]][1],
                    ]
                    + score
                    + score_new
                )
                scores = list(map(lambda x: x[0] + x[1], zip(scores, score)))
                scores_new = list(
                    map(lambda x: x[0] + x[1], zip(scores_new, score_new))
                )
            if _cnt:
                scores = [x / _cnt for x in scores]
                scores_new = [x / _cnt for x in scores_new]
                all_scores = scores + scores_new
                writer.writerow(["Average", "-", "-", "-", "-", "-"] + all_scores)
                if file == "Segmentation.csv":
                    request_average_scores = all_scores
                    request_cnt = _cnt
                elif file == "Segmentation_refine.csv":
                    request_refine_average_scores = all_scores
                elif file == "ResponseSegmentation.csv":
                    response_average_scores = all_scores
                    response_cnt = _cnt
                elif file == "ResponseSegmentation_refine.csv":
                    response_refine_average_scores = all_scores
    with open(os.path.join(sys.argv[4], "all.perf.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Tech." "Correctness (Cover or In Field Num. / Inferred Field Num.)",
                "Perfection (Accurate Field Num. / Correct Field Num.)",
                "Precision",
                "Recall",
                "Accuracy",
                "F1",
                "FPR",
            ]
        )
        writer.writerow(
            ["Average"]
            + [
                x / (response_cnt + request_cnt)
                for x in list(
                    map(
                        lambda x: x[0] * request_cnt + x[1] * response_cnt,
                        zip(request_average_scores, response_average_scores),
                    )
                )
            ]
        )

        writer.writerow(
            ["Average_refine"]
            + [
                x / (response_cnt + request_cnt)
                for x in list(
                    map(
                        lambda x: x[0] * request_cnt + x[1] * response_cnt,
                        zip(
                            request_refine_average_scores,
                            response_refine_average_scores,
                        ),
                    )
                )
            ]
        )
