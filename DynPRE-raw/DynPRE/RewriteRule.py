# -*- coding: utf-8 -*
from typing import List, Tuple

import Constants


class Location:
    def __init__(
        self,
        message_index: int,
        diection: Constants.DIRECTION,
        byte_region: Tuple[int, int],
    ) -> None:  # have not check response or request
        self.message_index = message_index
        self.direction = diection
        self.byte_region = byte_region

    def __str__(self) -> str:
        if self.direction == Constants.DIRECTION.REQUEST:
            return f"Req@{self.message_index}:{self.byte_region}"
        else:
            return f"Rsp@{self.message_index}:{self.byte_region}"

    def to_dict(self):
        return {
            "message_index": self.message_index,
            "direction": "rsp"
            if self.direction == Constants.DIRECTION.RESPONSE
            else "req",
            "byte_region": self.byte_region,
        }


class RewriteRule:
    def __init__(
        self, source: Location, references: List[Location], constraint: int
    ) -> None:
        self.source = source
        self.references = references
        self.constraint = constraint

    def __str__(self) -> str:
        return (
            f"RewriteRule: {self.source} -> ["
            + ",".join([str(reference) for reference in self.references])
            + "],"
            + f"constraint: {Constants.CONSTRAINT_TO_NAME[self.constraint]}"
        )

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "references": [reference.to_dict() for reference in self.references],
            "constraint": Constants.CONSTRAINT_TO_NAME[self.constraint],
        }
