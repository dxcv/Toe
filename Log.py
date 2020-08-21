#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Date   ：2020/8/21 23:14
@Author ：wfh1300
"""
import sys
import logging
import os
# https://www.cnblogs.com/sunsky303/p/9562300.html
class Logger:
    def __init__(self, filename="./Log/test.log"):
        self.logger = logging.getLogger("simple_logger")
        self.logger.setLevel(logging.DEBUG)

        # 不能将./Log/test.log直接作为文件路径，需要提前创建好./Log 文件夹
        if not os.path.exists("./Log"):
            os.mkdir("./Log")

        logging.basicConfig(filename=filename, level=logging.DEBUG)

        self.handler = logging.StreamHandler(sys.stdout)
        self.handler.setLevel(logging.DEBUG)

        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(self.formatter)

        self.logger.addHandler(self.handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == "__main__":
    lg = Logger()
    lg.error("haha")
