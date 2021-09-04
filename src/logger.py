#!/usr/bin/env python
# coding=utf-8
import logging
import tqdm
import sys
import prettytable

#class TqdmLoggingHandler(logging.Handler):
#    def __init__(self, level=logging.NOTSET):
#        super(self.__class__, self).__init__(level)
#
#    def emit(self, record):
#        try:
#            msg = self.format(record)
#            tqdm.tqdm.write(msg)
#            self.flush()
#        except (KeyboardInterrupt, SystemExit):
#            raise
#        except:
#            self.handleError(record)

class DummyTqdmFile:
    def __init__(self, file):
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, 'flush', lambda: None)()

# Setup logging
logging.basicConfig(
    #format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=DummyTqdmFile(sys.stderr)
)

logger = logging.getLogger(__name__)

