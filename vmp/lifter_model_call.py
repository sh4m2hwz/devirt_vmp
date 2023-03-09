from miasm.ir.analysis import LifterModelCall
from miasm.arch.vmp.sem import Lifter_vmp

class LifterModelCallVmp(Lifter_vmp, LifterModelCall):

    def __init__(self, loc_db):
        Lifter_vmp.__init__(self, loc_db)
    def get_out_regs(self, _):
        return set([self.sp])

    def sizeof_char(self):
        return 8

    def sizeof_short(self):
        return 16

    def sizeof_int(self):
        return 32

    def sizeof_long(self):
        return 32

    def sizeof_longlong(self):
        return 64

    def sizeof_pointer(self):
        return 64
