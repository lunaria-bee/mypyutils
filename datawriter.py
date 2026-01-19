import csv
import logging
from pathlib import Path


class DataWriter:
    '''TODO'''

    def __init__(
            self,
            path,
            ivars=(),
            dvars=(),
    ):
        '''TODO'''
        self.path = Path(path)
        self.ivars = tuple(ivars)
        self.dvars = tuple(dvars)

        if self.path.is_file():
            logging.debug(f"{repr(self.path)} exists")
            with open(self.path, 'r') as f:
                self.completed = set(
                    tuple(row[var] for var in self.ivars)
                    for row in csv.DictReader(f)
                )
        else:
            logging.debug(f"creating {repr(self.path)}")
            with open(self.path, 'w') as f:
                csv.DictWriter(f, self.ivars + self.dvars).writeheader()
            self.completed = set()

        logging.debug(f"self.completed={self.completed}")

    def write(self, rowdict):
        '''TODO'''
        varnames = self.ivars + self.dvars
        logging.debug("{{{}}}".format(
            "\n".join(
                f"{repr(varname)}: {repr(rowdict[varname])}"
                for varname in varnames
            )
        ))
        with open(self.path, 'a') as f:
            csv.DictWriter(f, varnames).writerow(rowdict)
