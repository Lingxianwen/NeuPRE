# -*- coding: utf-8 -*
import logging


class StraightTransformer:
    def __init__(self) -> None:
        pass

    def send_transform(
        self, request: bytes, sequence_num: int
    ) -> bytes:  # sequence_num start from 0
        return request

    def recv_transform(
        self, response: bytes, sequence_num: int, transform=False
    ) -> bytes:
        return response
