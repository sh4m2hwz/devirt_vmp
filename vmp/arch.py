from miasm.core.cpu import *
from miasm.core.utils import Disasm_Exception
from miasm.expression.expression import ExprId, ExprInt, ExprLoc, \
    ExprMem, ExprOp, is_expr
from miasm.core.asm_ast import AstId, AstMem

from miasm.arch.vmp.regs import *
import miasm.arch.vmp.regs as vmp_regs_module

class instruction_vmp(instruction):
  # Default delay slot
  # Note:
  #   - mandatory for the miasm Machine
  delayslot = 0

  def __init__(self, name, mode, args, additional_info=None):
    self.name = name
    self.mode = mode
    self.args = args
    self.additional_info = additional_info
    self.offset = None
    self.l = None
    self.b = None


  @staticmethod
  def arg2str(expr, pos=None, loc_db=None):
      """Convert mnemonics arguments into readable strings according to the
      vmp architecture and their internal types
      Notes:
          - it must be implemented ! However, a simple 'return str(expr)'
            could do the trick.
          - it is used to mimic objdump output
      Args:
          expr: argument as a miasm expression
          pos: position index in the arguments list
      """

      if isinstance(expr, ExprId) or isinstance(expr, ExprInt):
          return str(expr)

      elif isinstance(expr, ExprLoc):
          if loc_db is not None:
              return loc_db.pretty_str(expr.loc_key)
          else:
              return str(expr)
      # Raise an exception if the expression type was not processed
      message = "instruction_vmp.arg2str(): don't know what \
                  to do with a '%s' instance." % type(expr)
      raise Disasm_Exception(message)

  def __str__(self):
    """Return the mnemonic as a string.
    Note:
        - it is not mandatory as the instruction class already implement
          it. It used to get rid of the padding between the opcode and the
          arguments.
        - most of this code is copied from miasm/core/cpu.py
    """
    o = "%s" % self.name
    args = []
    if self.args:
        o += " "
    for i, arg in enumerate(self.args):
        if not is_expr(arg):
            raise ValueError('zarb arg type')
        x = self.arg2str(arg, pos=i)
        args.append(x)
    o += self.gen_args(args)
    return o

  def breakflow(self):
    """Instructions that stop a basic block."""
    if self.name in ["VMSWITCH"]:
      return True
    return False

  def splitflow(self):
    """Instructions that splits a basic block, i.e. the CPU can go somewhere else."""
    if self.name in ["VMSWITCH"]:
      return True
    return False

  def dstflow(self):
    """Instructions that explicitly provide the destination."""
    if self.name in ["VMSWITCH"]:
      return True
    return False

  def dstflow2label(self, loc_db):
    """Set the label for the current destination.
        Note: it is used at disassembly"""
    
  def getdstflow(self, loc_db):
    """Get the argument that points to the instruction destination."""
    addr = int(self.offset)
    if self.name == "VMSWITCH":
      return []
    loc_key = loc_db.get_or_create_offset_location(addr)
    return [ExprLoc(loc_key, 64)]
    
  def is_subcall(self):
    """Instructions used to call sub functions.
      vmp Does not have calls.
    """
    return False

class vmp_additional_info(object):
  """Additional vmp instructions information
  """

  def __init__(self):
    self.except_on_instr = False

class mn_vmp(cls_mn):
  # Define variables that stores information used to disassemble & assemble
  # Notes: - these variables are mandatory
  #        - they could be moved to the cls_mn class

  num = 0  # holds the number of mnemonics

  all_mn = list()  # list of mnenomnics, converted to metamn objects

  all_mn_mode = defaultdict(list) # mneomnics, converted to metamn objects
                                  # Note:
                                  #   - the key is the mode # GV: what is it ?
                                  #   - the data is a list of
                                  #     mnemonics

  all_mn_name = defaultdict(list)    # mnenomnics strings
                                     # Note:
                                     #   - the key is the mnemonic string
                                     #   - the data is the corresponding
                                     #     metamn object

  all_mn_inst = defaultdict(list)    # mnemonics objects
                                     # Note:
                                     #   - the key is the mnemonic Python class
                                     #   - the data is an instantiated
                                     #     object
  bintree = dict()  # Variable storing internal values used to guess a
                      # mnemonic during disassembly

  # Defines the instruction set that will be used
  instruction = instruction_vmp

  # Python module that stores registers information
  regs = vmp_regs_module

  max_instruction_len = 9
  # Default delay slot
  # Note:
  #   - mandatory for the miasm Machine
  delayslot = 0

  # Architecture name
  name = "vmp"

  # PC name depending on architecture attributes (here, l or b)
  # pc = PC

  def additional_info(self):
    """Define instruction side effects # GV: not fully understood yet
    When used, it must return an object that implements specific
    variables, such as except_on_instr.
    Notes:
        - it must be implemented !
        - it could be moved to the cls_mn class
    """

    return vmp_additional_info()

  @classmethod
  def gen_modes(cls, subcls, name, bases, dct, fields):
    """Ease populating internal variables used to disassemble & assemble, such
    as self.all_mn_mode, self.all_mn_name and self.all_mn_inst
    Notes:
        - it must be implemented !
        - it could be moved to the cls_mn class. All miasm architectures
          use the same code
    Args:
        cls: ?
        sublcs:
        name: mnemonic name
        bases: ?
        dct: ?
                fields: ?
    Returns:
        a list of ?
    """

    dct["mode"] = None
    return [(subcls, name, bases, dct, fields)]

  @classmethod
  def getmn(cls, name):
    """Get the mnemonic name
    Notes:
        - it must be implemented !
        - it could be moved to the cls_mn class. Most miasm architectures
          use the same code
    Args:
        cls:  the mnemonic class
        name: the mnemonic string
    """

    return name.upper()

  @classmethod
  def getpc(cls, attrib=None):
    """"Return the ExprId that represents the Program Counter.
    Notes:
        - mandatory for the symbolic execution
        - PC is defined in regs.py
    Args:
        attrib: architecture dependent attributes (here, l or b)
    """

    return PC

  @classmethod
  def getsp(cls, attrib=None):
    """"Return the ExprId that represents the Stack Pointer.
    Notes:
        - mandatory for the symbolic execution
        - SP is defined in regs.py
    Args:
        attrib: architecture dependent attributes (here, l or b)
    """

    return SP

  @classmethod
  def getbits(cls, bitstream, attrib, start, n):
    """Return an integer of n bits at the 'start' offset
        Note: code from miasm/arch/mips32/arch.py
    """

    # Return zero if zero bits are requested
    if not n:
      return 0
    o = 0  # the returned value
    while n:
      # Get a byte, the offset is adjusted according to the endianness
      offset = start // 8  # the offset in bytes
      # n_offset = cls.endian_offset(attrib, offset)  # the adjusted offset
      c = cls.getbytes(bitstream, offset, 1)
      if not c:
          raise IOError

      # Extract the bits value
      c = ord(c)
      r = 8 - start % 8
      c &= (1 << r) - 1
      l = min(r, n)
      c >>= (r - l)
      o <<= l
      o |= c
      n -= l
      start += l

    return o

  @classmethod
  def endian_offset(cls, attrib, offset):
    if attrib == "l":
      return offset
    else:
      raise NotImplementedError("bad attrib")

  def value(self, mode):
    v = super(mn_vmp, self).value(mode)
    if mode == 'l':
      return [x for x in v]
    else:
      raise NotImplementedError("bad attrib")

def addop(name, fields, args=None, alias=False):
  """
  Dynamically create the "name" object
  Notes:
      - it could be moved to a generic function such as:
        addop(name, fields, cls_mn, args=None, alias=False).
      - most architectures use the same code
  Args:
      name:   the mnemonic name
      fields: used to fill the object.__dict__'fields' attribute # GV: not understood yet
      args:   used to fill the object.__dict__'fields' attribute # GV: not understood yet
      alias:  used to fill the object.__dict__'fields' attribute # GV: not understood yet
  """

  namespace = {"fields": fields, "alias": alias}

  if args is not None:
      namespace["args"] = args

  # Dynamically create the "name" object
  type(name, (mn_vmp,), namespace)

class vmp_arg(m_arg):
  def asm_ast_to_expr(self, arg, loc_db):
    """Convert AST to expressions
        Note: - code inspired by miasm/arch/mips32/arch.py"""

    if isinstance(arg, AstId):
      if isinstance(arg.name, ExprId):
        return arg.name
      if isinstance(arg.name, str) and arg.name in gpr_names:
        return None  # GV: why?
      loc_key = loc_db.get_or_create_name_location(arg.name.encode())
      return ExprLoc(loc_key, 64)

    elif isinstance(arg, AstMem):
      addr = self.asm_ast_to_expr(arg.ptr, loc_db)
      if addr is None:
        return None
      return ExprMem(addr, 64)

    elif isinstance(arg, AstInt):
      return ExprInt(arg.value, 64)

    elif isinstance(arg, AstOp):
      args = [self.asm_ast_to_expr(tmp, loc_db) for tmp in arg.args]
      if None in args:
          return None
      return ExprOp(arg.op, *args)

    # Raise an exception if the argument was not processed
    message = "mep_arg.asm_ast_to_expr(): don't know what \
                to do with a '%s' instance." % type(arg)
    raise Exception(message)

class vmp_reg(reg_noarg, vmp_arg):
  """Generic vmp register
  Note:
      - the register size will be set using bs()
  """
  reg_info = gpr_infos  # the list of vmp registers defined in regs.py
  parser = reg_info.parser  # GV: not understood yet

class vmp_imm(imm_noarg, vmp_arg):
  """Generic vmp immediate
  Note:
      - the immediate size will be set using bs()
  """
  parser = base_expr
  def decode(self,v):
    v = self.decodeval(v)
    self.expr = ExprInt(v, 32)    
    return True

reg   = bs(l=8,  cls=(vmp_reg,),fname="reg")
imm64  = bs(l=64,sz=64,cls=(vmp_imm,vmp_arg),order=1,fname="imm64")

# mnemonics
addop("VPUSHI64", [bs8(0), imm64])
addop("VPUSHRQ", [bs8(1), reg])
addop("VPUSHVSP", [bs8(2), ])
addop("VLOADTOVSP", [bs8(3), ])
addop("VLOADQSTACK", [bs8(4), ])
addop("VLOADQ", [bs8(5), ])
addop("VPOPQ", [bs8(6), reg])
addop("VADDQ", [bs8(7), ])
addop("VANDNQ", [bs8(8),])
addop("VORNQ", [bs8(9), ])
addop("VMENTER", [bs8(10), ])
addop("VMSWITCH", [bs8(11),])