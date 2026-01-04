# -*- coding: utf-8 -*
import ast
import csv
import logging
import os
import sys
from typing import Dict, List, Tuple

import Constants

# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


# get zero ranges of a list
def get_zero_range(a: List) -> List[Tuple]:
    # Compute list "b" by replacing any non-zero value of "a" with 1
    b = list(map(int, [i != 0 for i in a]))

    # Compute ranges of 0 and ranges of 1, and only record ranges of 1
    idx = []  # result list of tuples
    start = 0  # index of first element of each range of zeros or non-zeros
    for n in range(len(b)):
        if (n + 1 == len(b)) or (b[n] != b[n + 1]):
            # Here: EITHER it is the last value of the list
            #       OR a new range starts at index n+1
            if not b[start]:
                idx.append((start, n + 1))
            start = n + 1

    return idx


# hexstream to bytes
def hexstream_to_bytes(hexstream: str):
    return bytes.fromhex(hexstream)


# Check if all zeros in the list appear consecutively
def check_zero_continuous(l: List[int]):
    zeros = 0
    for i in l:
        if i == 0:
            zeros += 1
    first_zero_index = -1
    last_zero_index = -2
    for i in range(len(l)):
        if l[i] == 0:
            if first_zero_index == -1:
                first_zero_index = i
            last_zero_index = i
    return zeros == last_zero_index - first_zero_index + 1
    # zero_to_no_zero_times = 0
    # no_zero_to_zero_times = 0
    # for i in range(len(l)):
    #     if l[i] == 0:
    #         if i == 0 or l[i - 1] != 0:
    #             no_zero_to_zero_times += 1
    #     else:
    #         if i > 0 and l[i - 1] == 0:
    #             zero_to_no_zero_times += 1
    # return no_zero_to_zero_times <= 1 and zero_to_no_zero_times <=1


class FieldIdentifier:
    def __init__(self, out_dir) -> None:
        self.out_dir = out_dir
        self.Requests = []
        self.RequestSegmentationIndexes = []
        self.Responses = []
        self.ResponseSegmentationIndexes = []
        self.varified_fields = []  # list of all fields that have been varified
        self.type_fields = []  # list of all fields that have been typed

        with open(
            os.path.join(self.out_dir, Constants.RequestSegmentationFile), "r"
        ) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for line in reader:
                self.RequestSegmentationIndexes.append(ast.literal_eval(line[1]))
                self.Requests.append(line[0])
        with open(
            os.path.join(self.out_dir, Constants.ResponseSegmentationFile), "r"
        ) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for line in reader:
                self.ResponseSegmentationIndexes.append(ast.literal_eval(line[1]))
                self.Responses.append(line[0])

    def add_varified_fields(
        self, field: Tuple[int, int]
    ):  # de-duplicate, while maintaining the join order and not overlapping with existing fields
        if field not in self.varified_fields:
            intersectant = False
            for f in self.varified_fields:
                if not (field[0] >= f[1] or field[1] <= f[0]):
                    intersectant = True
                    break
            if not intersectant:
                self.varified_fields.append(field)

    def add_type_fields(
        self, field: Tuple[int, int]
    ):  # De-duplicate while maintaining the join order
        if field not in self.type_fields:
            self.type_fields.append(field)

    def run(self):
        self.identify()
        logging.debug("varified fields: %s", self.varified_fields)
        logging.debug("type fields: %s", self.type_fields)
        self.refine_segmentation()

    def identify(self) -> Tuple[Tuple[int, int], Dict[str, List[str]]]:
        # Among all request segmentation results, the frequency of each segment was counted
        SegmentationIndexFrequency = {}
        for segmentation_index in self.RequestSegmentationIndexes:
            for i in range(len(segmentation_index) - 1):
                this_seg_index = (
                    segmentation_index[i],
                    segmentation_index[i + 1],
                )
                if this_seg_index not in SegmentationIndexFrequency:
                    SegmentationIndexFrequency[this_seg_index] = 1
                else:
                    SegmentationIndexFrequency[this_seg_index] += 1

        found = False  # Whether the category field is found
        times_upper_bound = None  # Start from the fields with more occurrences, then set the upper limit to the number of occurrences of those fields, and next time find the fields with less than that number
        while True:
            max_times = 0
            fields_to_check = []
            for k, v in SegmentationIndexFrequency.items():
                if v > max_times and (
                    times_upper_bound is None or v < times_upper_bound
                ):
                    max_times = v
                    fields_to_check = [k]
                elif v == max_times:
                    fields_to_check.append(k)
            if len(fields_to_check) == 0:
                break

            logging.debug(
                f"Checking type fields {fields_to_check}, their occurrence times is {max_times}"
            )

            for field in fields_to_check:
                is_type_field, (type_field, type_field_map) = self.check_type_field(
                    field
                )
                if is_type_field:
                    found = True
                    logging.info(
                        f"Type field is {type_field}, type field map is {type_field_map}"
                    )
                    self.add_type_fields(type_field)
            times_upper_bound = max_times
        if not found:
            logging.warning("No type field found")

    def check_type_field(
        self, type_field_index: Tuple[int, int]
    ) -> Tuple[bool, Tuple[Tuple[int, int], Dict[str, List[str]]]]:
        logging.debug(f"Checking type field {type_field_index}")

        request_type_field_map = {}
        for i in range(len(self.Requests)):
            if 2 * type_field_index[1] > len(self.Requests[i]) or 2 * type_field_index[
                1
            ] > len(self.Responses[i]):
                return False, (None, None)
            field_in_request = self.Requests[i][
                2 * type_field_index[0] : 2 * type_field_index[1]
            ]
            field_in_response = self.Responses[i][
                2 * type_field_index[0] : 2 * type_field_index[1]
            ]
            if field_in_request not in request_type_field_map:
                request_type_field_map[field_in_request] = [field_in_response]
            else:
                request_type_field_map[field_in_request].append(field_in_response)
        # de-duplication
        for field in request_type_field_map:
            request_type_field_map[field] = list(set(request_type_field_map[field]))

        logging.debug(
            f"request_type_field_map is {request_type_field_map}, length is {len(request_type_field_map)}"
        )

        each_to_one = True  # Each type_field corresponds to only one value
        for type_field in request_type_field_map:
            if len(request_type_field_map[type_field]) > 1:
                each_to_one = False
                break

        if each_to_one:
            if (
                len(request_type_field_map) != 1
                or len(list(request_type_field_map.keys())[0]) > 2
            ):  # If it is a constant field, make sure the field is longer than 1 (screen out fields like {'00': ['00']})
                self.add_varified_fields(type_field_index)
            if len(request_type_field_map) == 1:
                print(f"constant field: {type_field_index}", request_type_field_map)

            # Check if value -> type_field is one-to-one
            all_values = []
            for type_field in request_type_field_map:
                all_values.extend(request_type_field_map[type_field])
            all_values = list(set(all_values))

            if len(all_values) == len(request_type_field_map):
                if len(request_type_field_map) > 1 and len(
                    request_type_field_map
                ) != len(
                    self.Requests
                ):  # Sieve out constant fields, thus ensuring that there are at least two types in the flow, and remove fields like sequence numbers
                    return True, (type_field_index, request_type_field_map)
            return False, (None, None)
        else:
            # It is possible that the index range of the type field is larger than the actual one, so check it again after the clip.
            logging.debug(f"{type_field_index} does not meet requirements, try to clip")
            diff_mask = [0 for _ in range(type_field_index[1] - type_field_index[0])]
            for request_field, values in request_type_field_map.items():
                if len(values) > 1:
                    cur = hexstream_to_bytes(
                        values[0]
                    )  # cur compares each bit with each of the remaining values to see if they are different, using diff_mask to record
                    for i in range(1, len(values)):
                        for j in range(len(cur)):
                            diff_mask[j] |= cur[j] ^ hexstream_to_bytes(values[i])[j]
            assert not all(v == 0 for v in diff_mask)

            logging.debug(f"Diff mask is {diff_mask}")
            type_field_clips = get_zero_range(diff_mask)
            for type_field_clip in type_field_clips:
                is_type_field, (type_field, type_field_map) = self.check_type_field(
                    (
                        type_field_index[0] + type_field_clip[0],
                        type_field_index[0] + type_field_clip[1],
                    )
                )
                if (
                    is_type_field
                ):  # Currently it returns if one of the clips succeeds, or you can try all the clips
                    return is_type_field, (type_field, type_field_map)
            logging.debug(f"Clip {type_field_index} failed")
            return False, (None, None)

    def refine_varified_fields(self):
        refined_varified_fields = []
        for f in self.varified_fields:
            if len(refined_varified_fields) == 0:
                refined_varified_fields.append(f)
            else:
                ff = f
                intersectant = False
                for vf in refined_varified_fields:
                    if not (vf[0] >= ff[1] or vf[1] <= ff[0]):
                        intersectant = True
                if not intersectant:
                    refined_varified_fields.append(ff)
        self.varified_fields = refined_varified_fields

    def refine_segmentation(self):
        with open(
            os.path.join(self.out_dir, Constants.ResponseSegmentationRefinedFile), "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Hexstream", "Split Indexes", "Splited Hexstream"])
            for i in range(len(self.ResponseSegmentationIndexes)):
                refined_indexes = self.ResponseSegmentationIndexes[i]
                # add tuples in varified_fields to refined_indexes
                for vf in self.varified_fields:
                    refined_indexes.append(vf[0])
                    refined_indexes.append(vf[1])

                # refined_indexes de-duplicate and sort
                refined_indexes = sorted(list(set(refined_indexes)))
                # Re-segmentation
                message_hex = self.Responses[i]
                message_segmentation = " ".join(
                    message_hex[2 * refined_indexes[_] : 2 * refined_indexes[_ + 1]]
                    for _ in range(len(refined_indexes) - 1)
                )
                writer.writerow([message_hex, refined_indexes, message_segmentation])

        with open(
            os.path.join(self.out_dir, Constants.RequestSegmentationRefinedFile), "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Hexstream", "Split Indexes", "Splited Hexstream"])
            for i in range(len(self.RequestSegmentationIndexes)):
                refined_indexes = self.RequestSegmentationIndexes[i]
                for vf in self.type_fields:
                    refined_indexes.append(vf[0])
                    refined_indexes.append(vf[1])
                # refined_indexes de-duplicate and sort
                refined_indexes = sorted(list(set(refined_indexes)))

                message_hex = self.Requests[i]
                message_segmentation = " ".join(
                    message_hex[2 * refined_indexes[_] : 2 * refined_indexes[_ + 1]]
                    for _ in range(len(refined_indexes) - 1)
                )
                writer.writerow([message_hex, refined_indexes, message_segmentation])
