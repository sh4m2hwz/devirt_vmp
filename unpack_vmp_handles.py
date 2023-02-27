from triton import *
from dumpulator import Dumpulator
from unicorn import *
from unicorn.x86_const import *
import os
from sys import argv

def get_map_prot(name):
    if name == "UNDEFINED":
        return UC_PROT_NONE
    elif name == "PAGE_EXECUTE":
        return UC_PROT_EXEC
    elif name == "PAGE_EXECUTE_READ":
        return UC_PROT_READ | UC_PROT_EXEC
    elif name == "PAGE_EXECUTE_READWRITE":
        return UC_PROT_ALL
    elif name == "PAGE_READWRITE":
        return UC_PROT_READ | UC_PROT_WRITE
    elif name == "PAGE_READONLY":
        return UC_PROT_READ
    else:
        return UC_PROT_ALL

def load_binary(filename):
    dp = Dumpulator(filename,quiet=True)
    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    print("[*] loading dump into unicorn and triton dse engine")
    mappings = dp.memory.map()
    print("[*](unicorn) loading memory regions into unicorn")
    for m in mappings:
        prot = get_map_prot(m.protect.name)
        base = m.base
        size = m.region_size
        if prot == UC_PROT_NONE:
            continue
        if len(m.info) > 0:
            if m.info[0] == "PEB":
                print("[*](unicorn) loading peb into unicorn")
                mu.mem_map(0, size, prot)
                mu.mem_write(0,bytes(dp.memory.read(base,size)))
                continue
        print(f"[*](unicorn) mapping memory region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
        mu.mem_map(base, size, prot)
        print(f"[*](unicorn) writing data into region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
        data = bytes(dp.memory.read(base, size))
        mu.mem_write(base, data)
    print("[*](unicorn) initializing cpu registers")
    mu.reg_write(UC_X86_REG_RAX,dp.regs.rax)
    mu.reg_write(UC_X86_REG_RCX,dp.regs.rcx)
    mu.reg_write(UC_X86_REG_RDX,dp.regs.rdx)
    mu.reg_write(UC_X86_REG_RBX,dp.regs.rbx)
    mu.reg_write(UC_X86_REG_RSP,dp.regs.rsp)
    mu.reg_write(UC_X86_REG_RBP,dp.regs.rbp)
    mu.reg_write(UC_X86_REG_RSI,dp.regs.rsi)
    mu.reg_write(UC_X86_REG_RDI,dp.regs.rdi)
    mu.reg_write(UC_X86_REG_R8,dp.regs.r8)
    mu.reg_write(UC_X86_REG_R9,dp.regs.r9)
    mu.reg_write(UC_X86_REG_R10,dp.regs.r10)
    mu.reg_write(UC_X86_REG_R11,dp.regs.r11)
    mu.reg_write(UC_X86_REG_R12,dp.regs.r12)
    mu.reg_write(UC_X86_REG_R13,dp.regs.r13)
    mu.reg_write(UC_X86_REG_R14,dp.regs.r14)
    mu.reg_write(UC_X86_REG_R15,dp.regs.r15)
    print("[+] complete init execution context")
    return mu,dp.regs.rip

pop_regs = []
pco_reg = None
config = {"registers": [], "memory":[]}
stop = False
mem_modified = {}

def init_triton_dse(mu):
    print("[*](triton dse) init ctx")
    ctx = TritonContext(ARCH.X86_64)
    ctx.setMode(MODE.ALIGNED_MEMORY, True)
    ctx.setConcreteRegisterValue(ctx.registers.rip,mu.reg_read(UC_X86_REG_RIP))
    ctx.setConcreteRegisterValue(ctx.registers.rax,mu.reg_read(UC_X86_REG_RAX))
    ctx.setConcreteRegisterValue(ctx.registers.rcx,mu.reg_read(UC_X86_REG_RCX))
    ctx.setConcreteRegisterValue(ctx.registers.rdx,mu.reg_read(UC_X86_REG_RDX))
    ctx.setConcreteRegisterValue(ctx.registers.rbx,mu.reg_read(UC_X86_REG_RBX))
    ctx.setConcreteRegisterValue(ctx.registers.rsp,mu.reg_read(UC_X86_REG_RSP))
    ctx.setConcreteRegisterValue(ctx.registers.rbp,mu.reg_read(UC_X86_REG_RBP))
    ctx.setConcreteRegisterValue(ctx.registers.rsi,mu.reg_read(UC_X86_REG_RSI))
    ctx.setConcreteRegisterValue(ctx.registers.rdi,mu.reg_read(UC_X86_REG_RDI))
    ctx.setConcreteRegisterValue(ctx.registers.r8,mu.reg_read(UC_X86_REG_R8))
    ctx.setConcreteRegisterValue(ctx.registers.r9,mu.reg_read(UC_X86_REG_R9))
    ctx.setConcreteRegisterValue(ctx.registers.r10,mu.reg_read(UC_X86_REG_R10))
    ctx.setConcreteRegisterValue(ctx.registers.r11,mu.reg_read(UC_X86_REG_R11))
    ctx.setConcreteRegisterValue(ctx.registers.r12,mu.reg_read(UC_X86_REG_R12))
    ctx.setConcreteRegisterValue(ctx.registers.r13,mu.reg_read(UC_X86_REG_R13))
    ctx.setConcreteRegisterValue(ctx.registers.r14,mu.reg_read(UC_X86_REG_R14))
    ctx.setConcreteRegisterValue(ctx.registers.r15,mu.reg_read(UC_X86_REG_R15))
    print("[*](triton dse) mapping and writing datas into regions")
    for region in mu.mem_regions():
        print(f"[*](triton dse) mapping and writing data into region: va_range={hex(region[0])}-{hex(region[1])}")
        data = mu.mem_read(region[0],region[1]-region[0])
        ctx.setConcreteMemoryAreaValue(region[0], data)
    for address,value in mem_modified.items():
        print(f"[*] (triton dse) modifing after unicorn execution memory:",hex(address))
        ctx.setConcreteMemoryAreaValue(address, value)
    return ctx

is_stop_emu = False
ctx = None

def hook_insn(mu, address, size, _):
    global ctx
    global pco_reg
    global pop_regs
    global handle_address
    global stop
    global is_stop_emu
    insn = Instruction(address,bytes(mu.mem_read(address,size+1)))
    ctx.disassembly(insn)
    if is_stop_emu:
        is_stop_emu = False
        mu.emu_stop()
        return
    if len(pop_regs) == 16:
        print("[+] found vmexit\n[+] done emulation")
        stop = True
        is_stop_emu = True
        return
    opcode = insn.getType()
    if opcode == OPCODE.X86.RET:
        if not pco_reg:
            stop = True
            is_stop_emu = True
            return
        is_stop_emu = True
        return
    else:
        pco_reg = None
    try:
        op1 = insn.getOperands()[0]
    except:
        return
    if opcode == OPCODE.X86.JMP and op1.getType() == OPERAND.REG:
        is_stop_emu = True
        return
    elif opcode == OPCODE.X86.PUSH and op1.getType() == OPERAND.REG:
        pco_reg = op1
        for reg in pop_regs:
            if reg == op1.getName():
                pop_regs.remove(reg)
    elif opcode == OPCODE.X86.POP and op1.getType()  == OPERAND.REG:
        for reg in pop_regs:
            if reg == op1.getName():
                return
        pop_regs.append(op1.getName())
    elif opcode == OPCODE.X86.MOV:
        op1 = insn.getOperands()[0]
        op2 = insn.getOperands()[1]
        if op1.getType() == OPERAND.REG and op2.getType() == OPERAND.MEM:
            var_name = reg_name = op2.getBaseRegister().getName()
            config.get("registers").append({"reg_name":reg_name,"var_name":var_name})
            print("[+] append mem operands to config from insn ->",insn)
        elif op1.getType() == OPERAND.MEM and op2.getType() == OPERAND.REG:
            var_name = reg_name = op1.getBaseRegister().getName()
            config.get("registers").append({"reg_name":reg_name,"var_name":var_name})
            print("[+] append mem operands to config from insn ->",insn)
    elif opcode == OPCODE.X86.XOR:
        op1 = insn.getOperands()[0]
        op2 = insn.getOperands()[1]
        if op1.getType() == OPERAND.MEM and op2.getType() == OPERAND.REG:
            var_name = reg_name = op1.getBaseRegister().getName()
            config.get("registers").append({"reg_name":reg_name,"var_name":var_name})
            print("[+] append mem operands to config from insn ->",insn)

def hook_mem_access(mu, access, address, size, value, ctx):
    if access == UC_MEM_WRITE:
        mem_modified[address] = bytes(mu.mem_read(address,size))

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

import hashlib

def unpack(filenamedump):
    global stop
    global config
    global mem_modified
    global ctx
    mu, rip = load_binary(filenamedump)
    mu.reg_write(UC_X86_REG_RIP,rip)
    mu.hook_add(UC_HOOK_CODE, hook_insn, None)
    mu.hook_add(UC_HOOK_MEM_WRITE, hook_mem_access, None)    
    count = 0
    while not stop:
        ctx = init_triton_dse(mu)
        mem_modified = {}
        print("[+] start unicorn emulation")
        mu.emu_start(begin=mu.reg_read(UC_X86_REG_RIP), until=-1)
        print("[+] emulation done")
        symbolizing_vm_context(ctx,config)
        config = {"registers": [], "memory":[]}       
        emulate(ctx)
        trace = deobfuscate(ctx)
        trace_s = ""
        for insn in trace:
            trace_s += insn+"\n"
        md5 = hashlib.md5()
        md5.update(trace_s.encode())
        is_skip = False
        for dir in os.listdir("."):
            if "hdl" not in dir:
                continue
            if not os.path.isdir(dir):
                continue
            md5_target = hashlib.md5()
            md5_target.update(open(f"{dir}/hdl.asm",'rb').read())
            if md5_target.hexdigest() == md5.hexdigest():
                is_skip = True
                break
        if is_skip:
            is_skip=False
            print("[+] found already exists handler on disk, skipping")
        else:
            os.mkdir(f"hdl{count}")
            open(f"hdl{count}/hdl.asm","w").write(trace_s)
            print(f"dropped: hdl{count}/hdl.asm")
            count+=1
    print("finish unpacking handlers")

def main():
    unpack(argv[1])

if __name__ == "__main__":
    main()
