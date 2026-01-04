# -*- coding: utf-8 -*
from enum import Enum, IntEnum, unique

from StraightTransformer import StraightTransformer


@unique
class DIRECTION(IntEnum):
    REQUEST = 0
    RESPONSE = 1


SESSION_INTERVAL_SHORT = 0.1  # seconds
SESSION_INTERVAL_LONG = 0.5  # seconds

ResponseSegmentationFile = "ResponseSegmentation.csv"
RequestSegmentationFile = "Segmentation.csv"

ResponseSegmentationRefinedFile = "ResponseSegmentation_refine.csv"
RequestSegmentationRefinedFile = "Segmentation_refine.csv"

RuntimeRecordFile = "RuntimeRecord.txt"

RefineFields = "RefineFields.txt"

REWRITERULESRECORDFILE = "RewriteRulesRecord.json"

protocol_to_transformer = {"": StraightTransformer}

REWRITE_RULE_MAX_LEARN_TIMES = 7

LOG_FORMAT = "%(levelname)-6s %(filename)s:%(lineno)d:%(funcName)s %(message)s"

CONSTRAINTS = [
    lambda x: x,
    lambda x: x + 1,
    lambda x: 2 * x,
    lambda x: 2 * x + 1,
    lambda x: 3 * x,
    lambda x: 3 * x + 1,
    lambda x: 4 * x,
    lambda x: 4 * x + 1,
    lambda x: 5 * x,
    lambda x: 5 * x + 1,
]


CONSTRAINT_TO_NAME = [
    "y=x",
    "y=x+1",
    "y=2x",
    "y=2x+1",
    "y=3x",
    "y=3x+1",
    "y=4x",
    "y=4x+1",
    "y=5x",
    "y=5x+1",
]
