# -*- coding: utf-8 -*
import logging
from typing import List

import Constants
from RewriteRule import RewriteRule


class MessageRewriter:
    def __init__(self) -> None:
        self.rewrite_rules: List[RewriteRule] = []
        self.current_values = {}

    def display_rules(self):
        rules_str = "\n".join([str(rule) for rule in self.rewrite_rules])
        return rules_str

    def set_rules(self, rules: List[RewriteRule]):
        for rule in rules:
            assert rule.source.direction == Constants.DIRECTION.RESPONSE
        self.rewrite_rules = rules
        logging.debug(f"Set rewrite rules:\n{self.display_rules()}")

    def clear_rules(self):
        self.rewrite_rules.clear()

    def has_rule(self, sequence_num):
        for rule in self.rewrite_rules:
            if (
                rule.source.message_index == sequence_num
                and rule.source.direction == Constants.DIRECTION.RESPONSE
            ):
                return True
        return False

    def send_rewrite(
        self, request: bytes, sequence_num: int
    ) -> bytes:  # sequence_num start from 0
        rewrited_request = request
        for rule in self.rewrite_rules:
            for reference in rule.references:
                if (
                    reference.message_index == sequence_num
                    and reference.direction == Constants.DIRECTION.REQUEST
                ):
                    rewrited_request = (
                        rewrited_request[: reference.byte_region[0]]
                        + Constants.CONSTRAINTS[rule.constraint](
                            self.current_values[rule]
                        )
                        + rewrited_request[reference.byte_region[1] + 1 :]
                    )
                    logging.debug(
                        f"Rewrite live Req@{sequence_num}, region is {reference.byte_region}, value is {self.current_values[rule]}"
                    )
        return rewrited_request

    def recv_rewrite(self, response: bytes, sequence_num: int, rewrite=False) -> bytes:
        for rule in self.rewrite_rules:
            if (
                rule.source.message_index == sequence_num
                and rule.source.direction == Constants.DIRECTION.RESPONSE
            ):
                value = response[
                    rule.source.byte_region[0] : rule.source.byte_region[1] + 1
                ]
                self.current_values[rule] = value
                logging.debug(
                    f"Obtain value for {rule} in live Rsp@{sequence_num}, value is {value}"
                )

        return response
