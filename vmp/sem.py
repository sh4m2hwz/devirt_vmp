from miasm.arch.aarch64.sem import extend_arg
from miasm.core.cpu import sign_ext
from miasm.expression.expression import *
from miasm.arch.vmp.regs import *
from miasm.arch.vmp.arch import mn_vmp
from miasm.ir.ir import Lifter

def vpushi64(_, instr, val):
    e = [
        ExprAssign(SP,SP - ExprInt(8,64)),
        ExprAssign(ExprMem(SP,64),ExprInt(val.arg,64))
    ]
    return e, []

def vpushrq(_, instr, reg):
    e = [
        ExprAssign(SP,SP - ExprInt(8,64)),
        ExprAssign(ExprMem(SP,64),reg)
    ]
    return e, []

def vpopq(_, instr, reg):
    e = [
        ExprAssign(SP,ExprOp('+',SP,ExprInt(8,64))),
        ExprAssign(reg,ExprMem(SP,64))
    ]
    return e, []

def vpushvsp(_, instr):
    e = [
        ExprAssign(SP,SP - ExprInt(8,64)),
        ExprAssign(ExprMem(SP,64),SP)
    ]
    return e, []

def vloadtovsp(_, instr):
    e = [
        ExprAssign(ExprMem(SP,64),ExprMem(ExprMem(SP,64),64)),
        ExprAssign(SP,ExprMem(SP,64))
    ]
    return e, []

def vloadq(_, instr):
    e = [
        ExprAssign(ExprMem(SP,64),ExprMem(ExprMem(SP,64),64))
    ]
    return e,[]

def vloadqstack(_, instr):
    e = [
        ExprAssign(ExprMem(SP,64),ExprMem(ExprMem(SP,64),64))
    ]
    return e,[]

def vmenter(_, instr):
    e = [
        ExprAssign(SP,SP_init),
        ExprAssign(PC,PC_init)
    ]
    return e,[]

def vmswitch(_, instr):
    return [],[]

def vaddq(_, instr):
    res = ExprMem(SP,64) + ExprMem(ExprOp('+',SP,ExprInt(8,64)),64)
    e = [
        ExprAssign(ExprMem(ExprOp('+',SP,ExprInt(8,64)),64),res),
        ExprAssign(ExprMem(SP,64),ExprId('VADDQ_FLAGS',64))
    ]
    return e, []

def vornq(_ ,instr):
    res = ~(ExprMem(SP,64) | ExprMem(ExprOp('+',SP,ExprInt(8,64)),64))
    e = [
        ExprAssign(ExprMem(ExprOp('+',SP,ExprInt(8,64)),64),res),
        ExprAssign(ExprMem(SP,64),ExprId('VORNQ_FLAGS',64))
    ]
    return e, []

def vandnq(_ ,instr):
    res = ~(ExprMem(SP,64) & ExprMem(ExprOp('+',SP,ExprInt(8,64)),64))
    e = [
        ExprAssign(ExprMem(ExprOp('+',SP,ExprInt(8,64)),64),res),
        ExprAssign(ExprMem(SP,64),ExprId('VANDNQ_FLAGS',64))
    ]
    return e, []


mnemo_func = {
    "VPUSHI64":vpushi64,
    "VPUSHVSP":vpushvsp,
    "VPUSHRQ": vpushrq,
    "VLOADQSTACK": vloadqstack,
    "VLOADTOVSP": vloadtovsp,
    "VLOADQ": vloadq,
    "VPOPQ": vpopq,
    "VADDQ": vaddq,
    "VANDNQ": vandnq,
    "VORNQ": vornq,
    "VMENTER": vmenter,
    "VMSWITCH": vmswitch
}


class Lifter_vmp(Lifter):
  """Toshiba MeP miasm IR - Big Endian
      It transforms an instructon into an IR.
  """
  addrsize = 64
  def __init__(self, loc_db=None):
    Lifter.__init__(self, mn_vmp, None, loc_db)
    self.pc = mn_vmp.getpc()
    self.sp = mn_vmp.getsp()
    self.IRDst = ExprId("IRDst", 64)

  def get_ir(self, instr):
    """Get the IR from a miasm instruction."""
    args = instr.args
    instr_ir, extra_ir = mnemo_func[instr.name](self, instr, *args)
    return instr_ir, extra_ir
