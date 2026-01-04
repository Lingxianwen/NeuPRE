# -*- coding: utf-8 -*
import csv
import logging
import os
from typing import List, Tuple

import Constants
from helpers import *


class RecordWriter:
    def __init__(self, fold):
        self.fold = fold
        if not os.path.exists(self.fold):
            os.makedirs(self.fold)
        probe_record = open(os.path.join(self.fold, Constants.RuntimeRecordFile), "w")
        probe_record.close()
        refine_fields_file = open(os.path.join(self.fold, Constants.RefineFields), "w")
        refine_fields_file.close()
        rewrite_rules_file = open(
            os.path.join(self.fold, Constants.REWRITERULESRECORDFILE), "w"
        )
        rewrite_rules_file.close()
        with open(os.path.join(self.fold, Constants.RequestSegmentationFile), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Hexstream", "Split Indexes", "Splited Hexstream"])

        for filename in [
            Constants.ResponseSegmentationFile,
        ]:
            with open(os.path.join(self.fold, filename), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Hexstream", "Split Indexes", "Splited Hexstream"])

    def new_seed(self, seed_index):
        with open(os.path.join(self.fold, Constants.RuntimeRecordFile), "a") as f:
            f.writelines("========Trace " + str(seed_index) + "========\n")

    def write_segmentation(self, Trace, message_index):
        with open(os.path.join(self.fold, Constants.RequestSegmentationFile), "a") as f:
            writer = csv.writer(f)
            message_hex = Trace.M[message_index].raw["Content"].hex()
            seg_index = Trace.SG[message_index]
            assert len(message_hex) == seg_index[-1] * 2
            message_segmentation = " ".join(
                message_hex[2 * seg_index[_] : 2 * seg_index[_ + 1]]
                for _ in range(len(seg_index) - 1)
            )
            writer.writerow([message_hex, seg_index, message_segmentation])

    def write_responses_segmentation(self, ans, filename):
        with open(os.path.join(self.fold, filename), "a") as f:
            writer = csv.writer(f)
            for hexstream, seg_index in ans:
                message_hex = hexstream
                seg_index = seg_index
                assert len(message_hex) == seg_index[-1] * 2
                message_segmentation = " ".join(
                    message_hex[2 * seg_index[_] : 2 * seg_index[_ + 1]]
                    for _ in range(len(seg_index) - 1)
                )
                writer.writerow([message_hex, seg_index, message_segmentation])

    def write_probe_record(self, Trace, message_index):
        with open(os.path.join(self.fold, Constants.RuntimeRecordFile), "a") as f:
            f.writelines(
                "*** Message Index-" + str(message_index) + " ***\n"
            )  # write the message information
            message_hex = Trace.M[message_index].raw["Content"].hex()
            f.writelines("Content :" + message_hex + "\n")
            f.writelines("Original Response" + "\n")  # write the original response
            f.writelines(
                Trace.R[message_index].hex()
                + "\t"
                + ascii_dump(Trace.R[message_index])
                + "\n"
            )

            f.writelines("Probe Result:" + "\n")  # write the results of probe
            f.writelines("PI" + "\n")  # PI
            for n in Trace.PI[message_index]:
                f.write(str(n) + " ")
            f.writelines("\n")
            for index, n in enumerate(Trace.PI[message_index]):
                f.write(
                    str(n) + "(" + message_hex[index * 2 : index * 2 + 2] + ")" + " "
                )
            f.writelines("\n")
            f.writelines("PR and PS" + "\n")
            for n in range(len(Trace.PR[message_index])):
                f.writelines(
                    "("
                    + str(n)
                    + ") "
                    + Trace.PR[message_index][n].hex()
                    + "\t"
                    + ascii_dump(Trace.PR[message_index][n])
                    + "\n"
                )
                f.writelines(str(Trace.PS[message_index][n]) + "\n")
            f.writelines("PRA" + "\n")
            for n in range(len(Trace.PRA[message_index])):
                f.writelines(
                    (
                        ("( %2d ) " % (n - 2))
                        + Trace.PRA[message_index][n].hex()
                        + "\t"
                        + ascii_dump(Trace.PRA[message_index][n])
                        + "\n"
                    )
                )

            f.writelines("\n")

    def write_record(self, Trace, message_index):
        self.write_probe_record(Trace, message_index)
        self.write_segmentation(Trace, message_index)

    def write_type_field(self, type_field_lst: List[Tuple[int, int]]):
        with open(os.path.join(self.fold, Constants.RefineFields), "a") as f:
            f.writelines("Type Fields:" + "\n")
            for n in type_field_lst:
                f.writelines(str(n) + "\n")

    def write_varified_field(self, varified_field_lst: List[Tuple[int, int]]):
        with open(os.path.join(self.fold, Constants.RefineFields), "a") as f:
            f.writelines("Varified Fields:" + "\n")
            for n in varified_field_lst:
                f.writelines(str(n) + "\n")

    def write_rewrite_rules(self, rewrite_rules_json):
        with open(os.path.join(self.fold, Constants.REWRITERULESRECORDFILE), "a") as f:
            f.writelines(rewrite_rules_json)
