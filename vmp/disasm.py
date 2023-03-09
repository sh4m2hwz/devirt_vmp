from miasm.core.asmblock import disasmEngine
from miasm.arch.vmp.arch import mn_vmp


class dis_vmp(disasmEngine):
  """MeP miasm disassembly engine - Big Endian
      Notes:
          - its is mandatory to call the miasm Machine
  """
  def __init__(self, bs=None, **kwargs):
    super(dis_vmp, self).__init__(mn_vmp, None, bs, **kwargs)