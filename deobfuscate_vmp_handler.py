from triton import *
from dumpulator import Dumpulator
from sys import argv
import json

def load_binary(filename):
    dp = Dumpulator(filename,quiet=True)
    print("[*] loading dump into triton dse engine")
    ctx = TritonContext(ARCH.X86_64)
    ctx.setMode(MODE.ALIGNED_MEMORY, True)
    mappings = dp.memory.map()
    for m in mappings:
        base = m.base
        size = m.region_size
        if m.protect.name == "UNDEFINED":
            continue
        if len(m.info) > 0:
            if m.info[0] == "PEB":
                print("[*] loading peb")
                ctx.setConcreteMemoryAreaValue(0,bytes(dp.memory.read(base, size)))
                continue
        print(f"[*] writing data into region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
        ctx.setConcreteMemoryAreaValue(base,bytes(dp.memory.read(base, size)))
    print("[*] initializing cpu registers")
    init_triton_dse_regs(dp, ctx)
    print("[+] complete init execution context")
    return ctx,dp.regs.rip


def init_triton_dse_regs(dp,ctx):
    ctx.setConcreteRegisterValue(ctx.registers.rax,dp.regs.rax)
    ctx.setConcreteRegisterValue(ctx.registers.rcx,dp.regs.rcx)
    ctx.setConcreteRegisterValue(ctx.registers.rdx,dp.regs.rdx)
    ctx.setConcreteRegisterValue(ctx.registers.rbx,dp.regs.rbx)
    ctx.setConcreteRegisterValue(ctx.registers.rsp,dp.regs.rsp)
    ctx.setConcreteRegisterValue(ctx.registers.rbp,dp.regs.rbp)
    ctx.setConcreteRegisterValue(ctx.registers.rsi,dp.regs.rsi)
    ctx.setConcreteRegisterValue(ctx.registers.rdi,dp.regs.rdi)
    ctx.setConcreteRegisterValue(ctx.registers.r8,dp.regs.r8)
    ctx.setConcreteRegisterValue(ctx.registers.r9,dp.regs.r9)
    ctx.setConcreteRegisterValue(ctx.registers.r10,dp.regs.r10)
    ctx.setConcreteRegisterValue(ctx.registers.r11,dp.regs.r11)
    ctx.setConcreteRegisterValue(ctx.registers.r12,dp.regs.r12)
    ctx.setConcreteRegisterValue(ctx.registers.r13,dp.regs.r13)
    ctx.setConcreteRegisterValue(ctx.registers.r14,dp.regs.r14)
    ctx.setConcreteRegisterValue(ctx.registers.r15,dp.regs.r15)
    

def symbolizing_vm_context(ctx,config):
    print("[+] symbolizing runtime context")
    sym_regs = config.get('registers')
    sym_mems = config.get("memory")
    if sym_regs == None:
        print("[-] invalid config: registers array not found!")
        exit(-1)
    if sym_mems == None:
        print("[-] invalid config: memory array not found!")
        exit(-1)
    for sym_reg in sym_regs:
        reg_name = sym_reg.get("reg_name")
        var_name = sym_reg.get("var_name")
        if reg_name == None or var_name == None:
            print("[-] invalid reg line in config")
            exit(-1)
        print(f"[+] symbolizing register {reg_name} as {var_name}")
        ctx.symbolizeRegister(ctx.getRegister(reg_name),var_name)
    for sym_mem in sym_mems:
        address = sym_mem.get("address")
        size = sym_mem.get("size")
        if address == None or size == None:
            print("[-] invalid mem line in config")
            exit(-1)
        a = MemoryAccess(int(address,16),int(size))
        print(f"symbolizing memory {a}")
        ctx.symbolizeMemory(a)
    print("[+] finish symbolize execution context")


def emulate(ctx):
    print("[*] starting symbolic emulation")
    rip = ctx.getConcreteRegisterValue(ctx.registers.rip)
    while True:
        opcodes = ctx.getConcreteMemoryAreaValue(rip,16)
        insn = Instruction(rip,opcodes)
        ctx.disassembly(insn)
        opcode = insn.getType()
        ops = insn.getOperands()
        if opcode == OPCODE.X86.TEST or opcode == OPCODE.X86.CMP:
            rip+=insn.getSize()
            continue
        ctx.processing(insn)
        if opcode == OPCODE.X86.RET or \
        opcode == OPCODE.X86.JMP and ops[0].getType() == OPERAND.REG:
            break
        rip = ctx.getConcreteRegisterValue(ctx.registers.rip)
    print("[+] done emulation")
    return

#def slice_trace(ctx,expr):
#    sliced_reg = ctx.sliceExpressions(expr)
#    keys = sorted(sliced_reg)
#    trace = {}
#    for key in keys:
#        disasm = sliced_reg[key].getDisassembly()
#        if len(disasm) == 0:
#            continue
#        address, insn = disasm.split(": ")
#        trace[int(address,16)] = insn
#    return trace


#def slice_mem_trace(ctx,mem):
#    return slice_trace(ctx,ctx.getSymbolicMemory(mem))

#def slice_reg_trace(ctx,reg):
#    return slice_trace(ctx,ctx.getSymbolicRegister(reg))


#def concat_traces(arr_traces):
#    trace_addrs = []
#    for trace in arr_traces:
#        trace_addrs+=list(trace.keys())
#    concat_trace = {}
#    for address in sorted(trace_addrs):
#        for trace in arr_traces:
#            if trace.get(address) != None:
#                concat_trace[address] = trace.get(address)
#    return concat_trace

def deobfuscate(ctx):
    dup_insns = []
    sexprs = ctx.getSymbolicExpressions()
    keys = sorted(sexprs)
    for key in keys:
        ty = sexprs[key].getType()
        if ty == SYMBOLIC.REGISTER_EXPRESSION or \
        ty == SYMBOLIC.VOLATILE_EXPRESSION or \
        ty == SYMBOLIC.MEMORY_EXPRESSION:
            disasm = sexprs[key].getDisassembly()
            if len(disasm) > 0:
                dup_insns.append(disasm)
    insns = []
    for i in range(len(dup_insns)):
        if dup_insns[i] not in insns:
            insns.append(dup_insns[i])
    return insns


def parse_config(filenameconfig):
    print("[*] parsing config")
    with open(filenameconfig) as f:
        j = json.load(f)
        return j


def drop_hdl_insn(filenamedump,filenameconfig):
    ctx, rip = load_binary(filenamedump)
    ctx.setConcreteRegisterValue(ctx.registers.rip,rip)
    config = parse_config(filenameconfig)
    symbolizing_vm_context(ctx,config)
    emulate(ctx)
    print("[+] clearing obfuscated vmp handler")
    trace = deobfuscate(ctx)
    print("[+] complete deobfuscate")
    trace_s = str()
    for insn in trace:
        trace_s+=insn.split(": ")[1]+"\n"
    return trace_s

def main():
    if len(argv) != 3:
        print("""
./vmp_deobfuscate_handler.py <dump file> <symconfig.json>
<dump file> - dump file contains EP vmp handler
<symconfig.json> - config contains symbolic registers and memory        
""")
    else:
        open("deobfuscated_vmp_hdl.asm","w").write(drop_hdl_insn(argv[1],argv[2]))
        print("[+] saved deobfuscated instructions handler trace in file deobfuscated_vmp_hdl.asm")

if __name__ == "__main__":
    main()
