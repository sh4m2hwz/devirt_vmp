from triton import *
from unicorn import *
from unicorn.x86_const import *
from dumpulator import Dumpulator
import os
from sys import argv

def check_in_region(vmp_range,address):
    if vmp_range[0] <= address and address <= vmp_range[1]:
        return True
    else:
        return False

def get_reg(reg):
    if not reg:
        return 0
    elif reg == ctx.registers.rax:
        return UC_X86_REG_RAX
    elif reg == ctx.registers.rcx:
        return UC_X86_REG_RCX
    elif reg == ctx.registers.rdx:
        return UC_X86_REG_RDX
    elif reg == ctx.registers.rbx:
        return UC_X86_REG_RBX
    elif reg == ctx.registers.rsp:
        return UC_X86_REG_RSP
    elif reg == ctx.registers.rbp:
        return UC_X86_REG_RBP
    elif reg == ctx.registers.rsi:
        return UC_X86_REG_RSI
    elif reg == ctx.registers.rdi:
        return UC_X86_REG_RDI
    elif reg == ctx.registers.r8:
        return UC_X86_REG_R8
    elif reg == ctx.registers.r9:
        return UC_X86_REG_R9
    elif reg == ctx.registers.r10:
        return UC_X86_REG_R10
    elif reg == ctx.registers.r11:
        return UC_X86_REG_R11
    elif reg == ctx.registers.r12:
        return UC_X86_REG_R12
    elif reg == ctx.registers.r13:
        return UC_X86_REG_R13
    elif reg == ctx.registers.r14:
        return UC_X86_REG_R14
    elif reg== ctx.registers.r15:
        return UC_X86_REG_R15

def get_target_addr_call(mu,op):
    if op.getType() == OPERAND.MEM:
        base = mu.reg_read(get_reg(op.getBaseRegister()))
        index = mu.reg_read(get_reg(op.getIndexRegister()))
        scale = mu.reg_read(get_reg(op.getScaleIndex()))
        disp = op.getDisplacement().getValue()
        address = base+index*scale+disp
    elif op.getType() == OPERAND.REG:
        address = mu.reg_read(get_reg(op))
    elif op.getType() == OPERAND.IMM:
        address = op.getValue()
    else:
        print("cannot disassembly at address:",address)
        exit(-1)
    return address

is_stop = False

def hook_insn(mu, address, size, hook_ctx):
    global is_stop
    if is_stop:
        is_stop = False
        mu.emu_stop()
        return
    ctx = hook_ctx['ctx']
    vmp_range = hook_ctx['vmp_range']
    bb = hook_ctx['bb']
    insn = Instruction(address,bytes(mu.mem_read(address,size+1)))
    ctx.disassembly(insn)
    if insn.getType() == OPCODE.X86.CALL:
        bb.add(insn)
        op = insn.getOperands()[0]
        target_call_address = get_target_addr_call(mu,op)
        if not check_in_region(vmp_range, target_call_address):
            print("[+] found external call:",insn)
            mu.emu_stop()
            exit(-1)
        else:
            is_stop = True
    elif insn.getType() == OPCODE.X86.JMP \
    and insn.getOperands()[0].getType() == OPERAND.REG:
        bb.add(insn)
        is_stop = True
    elif insn.getType() == OPCODE.X86.RET:
        bb.add(insn)
        is_stop = True
    elif not insn.isControlFlow():
        bb.add(insn)
        

class VmpAnalyzerX64:
    def __init__(self, filenamedump,vmp_segment_range={"start":0,"end":0}):
        self.filenamedump = filenamedump
        self.dp = None
        self.mu = None
        self.ctx = None
        self.VIP = None
        self.VSP = None
        self.reg_pco = None
        self.RKEY = None
        self.VREGISTERS = None
        self.is_find_vmenter = False
        self.bb_vmenter = None
        self.__next_bb_address = 0
        self.vmp_segment_range = [vmp_segment_range['start'],vmp_segment_range['end']]
        if self.vmp_segment_range[0] == 0 and self.vmp_segment_range[1] == 0:
            print("[-] please pass valid vmp segment range")
            return None
        self.__load_binary()
        self.__init_triton_dse()
        self.symbolizeRegisters()
        self.hook_ctx = {"vmp_range":self.vmp_segment_range,"ctx":self.ctx,'bb':None}
        self.mu.hook_add(UC_HOOK_CODE,hook_insn,self.hook_ctx)
        self.entry_point = self.dp.regs.rip
    #
    def analyzeBasicBlock(self, address):
        self.hook_ctx['bb'] = BasicBlock([])
        self.mu.emu_start(address,-1)
        self.__next_bb_address = self.mu.reg_read(UC_X86_REG_RIP)
        return self.hook_ctx['bb']
    #
    def getNextEntryBasicBlock(self):
        return self.__next_bb_address
    #
    def is_stack(self,ast_variables):
        for var in ast_variables:
            if var.getSymbolicVariable().getAlias() == "rsp":
                return True
        return False
    #
    def find_vmenter(self, bb):
        astctx = self.ctx.getAstContext()
        saved_regs = []
        for insn in bb.getInstructions():
            if len(saved_regs) == 16:
                print("[+] found vmenter at address: ",hex(bb.getFirstAddress()))
                return True
            elif insn.isSymbolized() and insn.getType() == OPCODE.X86.PUSHFQ:
                if "eflags" in saved_regs:
                    continue
                saved_regs.append('eflags')
            elif insn.isSymbolized() and insn.getType() == OPCODE.X86.PUSH and insn.getOperands()[0].getType() == OPERAND.REG:
                reg = insn.getOperands()[0]
                if reg.getName() in saved_regs:
                    continue
                saved_regs.append(reg.getName())
        if len(saved_regs) == 16:
            print("[+] found vmenter at address: ",hex(bb.getFirstAddress()))
            return True
        else:
            return False
    #
    def analyze_vmenter(self):  
        for insn in self.bb_vmenter.getInstructions():
            if insn.getType() == OPCODE.X86.MOV:
                ops = insn.getOperands()
                op1 = ops[0]
                op2 = ops[1]
                if op1.getType() == OPERAND.REG \
                and op2.getType() == OPERAND.MEM:
                    base = op2.getBaseRegister()
                    disp = op2.getDisplacement()
                    if base != None and disp != None \
                    and base == self.ctx.registers.rsp and disp.getValue() == 0x90:
                        print("[+] found MOV reg, [RSP+90], calc VIP:",insn)
                        self.VIP=op1
                        print("[*] mapped reg VIP is:",ops[0].getName())
            if insn.getType() == OPCODE.X86.MOV:
                ops = insn.getOperands()
                if ops[0].getType() == OPERAND.REG \
                and ops[1].getType() == OPERAND.REG \
                and ops[1] == self.ctx.registers.rsp:
                    print("[+] found MOV contains VSP:",insn)
                    self.VSP=ops[0]
                    self.VREGISTERS=ops[1]
                    print("[*] mapped reg VSP is:",ops[0].getName())
                    print("[*] mapped reg VREGISTERS is:",ops[1].getName())
            elif insn.getType() == OPCODE.X86.POP:
                ops = insn.getOperands()
                if ops[0].getType() == OPERAND.REG:
                    print("[+] found POP contains RKEY:",insn)
                    self.RKEY = ops[0]
            elif insn.getType() == OPCODE.X86.PUSH:
                self.reg_pco = insn.getOperands()[0]
            elif insn.getType() == OPCODE.X86.RET:
                print("[+] found PUSH REG; RET pattern")
                print("[+] reg pco (Path Constraint) is:",self.reg_pco.getName())
                break
            elif insn.getType() == OPCODE.X86.JMP \
            and insn.getOperands()[0].getType() == OPERAND.REG:
                print("[+] found JMP REG pattern")
                self.reg_pco = insn.getOperands()[0]
                print("[+] reg pco (Path Contraint) is:",self.reg_pco.getName())
                break
        if not self.reg_pco or not self.RKEY or not self.VSP \
        or not self.VREGISTERS or not self.VIP:
            print("[-] this devirtualizer support vmprotect 3.6.x, different version")
            exit(1)
        print("[+] complete finishing analyzing vmenter handler")
    #
    def analyze_vmswitch(self,bb):
        is_mov_reg_vsp = False
        is_mov_reg_mem_vsp = False
        reg_VIP_inter = None
        is_mov_new_vip = False
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG \
            and ops[1] == self.VSP:
                is_mov_reg_vsp = True
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM:
                mem = ops[1]
                base = mem.getBaseRegister()
                if base != 0 and base == self.VSP:
                    is_mov_reg_mem_vsp = True
                    reg_VIP_inter = ops[0]
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG:
                if reg_VIP_inter != None and ops[1] == reg_VIP_inter:
                    is_mov_new_vip = True
        if is_mov_new_vip and is_mov_reg_mem_vsp and is_mov_new_vip:
            print("[+] found vmswitch handler at address:",hex(bb.getFirstAddress()))
            return True
        return False
    #
    def analyze_vmexit(self,bb):
        pass
    #
    def analyze_vmops(self,bb):
        pass
    #
    def analyze_vmhandlers(self,bb):
        print("[*] analyze vmp handler\n[*] simplification basicblock..")
        bb_simp = self.ctx.simplify(bb)
        bb_simp = BasicBlock(
            [bb_simp.getInstructions()[-1]]
            +
            bb_simp.getInstructions()[:-1]
        )
        self.ctx.processing(bb_simp,bb.getFirstAddress())
        print("[+] simpplified handler:")
        print(bb_simp)
        print("[*] start classification handler..")
        if self.analyze_vmswitch(bb_simp):
            return True
        if self.analyze_vmexit(bb_simp):
            return True
        self.analyze_vmops(bb_simp)
        return False
    #
    def analyze(self):
        bb_entry = self.entry_point
        while True:
            bb = self.analyzeBasicBlock(bb_entry)
            self.ctx.processing(bb,bb.getFirstAddress())
            if not self.is_find_vmenter:
                is_find_vmenter = self.find_vmenter(bb)
                if is_find_vmenter:
                    self.is_find_vmenter = True
                    self.bb_vmenter = bb
                    self.analyze_vmenter()
            else:
                is_vmswitch_or_vmexit = self.analyze_vmhandlers(bb)
                if is_vmswitch_or_vmexit:
                    break
            bb_entry = self.getNextEntryBasicBlock()
    #
    def symbolizeRegisters(self):
        print("[*] symbolizing all registers")
        for reg in ['rax','rcx','rdx','rbx','rsp','rbp','rsi','rdi','r8','r9','r10','r11','r12','r13','r14','r15','rip','eflags']:
            self.ctx.symbolizeRegister(self.ctx.getRegister(reg),reg)
    #
    def __init_triton_dse(self):
        print("[*](triton dse) init ctx")
        self.ctx = TritonContext(ARCH.X86_64)
        self.ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self.ctx.setMode(MODE.CONSTANT_FOLDING, True)
        self.ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        print("[*](triton dse) mapping and writing datas into regions")
        for region in self.mu.mem_regions():
            print(f"[*](triton dse) mapping and writing data into region: va_range={hex(region[0])}-{hex(region[1])}")
            data = self.mu.mem_read(region[0],region[1]-region[0])
            self.ctx.setConcreteMemoryAreaValue(region[0], data)
    #
    def __get_map_prot(self,name):
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
    #
    def __load_binary(self):
        self.dp = Dumpulator(self.filenamedump,quiet=True)
        self.mu = Uc(UC_ARCH_X86, UC_MODE_64)
        print("[*] loading dump into unicorn and triton dse engine")
        mappings = self.dp.memory.map()
        print("[*](unicorn) loading memory regions into unicorn")
        for m in mappings:
            prot = self.__get_map_prot(m.protect.name)
            base = m.base
            size = m.region_size
            if prot == UC_PROT_NONE:
                continue
            if len(m.info) > 0:
                if m.info[0] == "PEB":
                    print("[*](unicorn) loading peb into unicorn")
                    self.mu.mem_map(0, size, prot)
                    self.mu.mem_write(0,bytes(self.dp.memory.read(base,size)))
                    continue
            print(f"[*](unicorn) mapping memory region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
            self.mu.mem_map(base, size, prot)
            print(f"[*](unicorn) writing data into region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
            data = bytes(self.dp.memory.read(base, size))
            self.mu.mem_write(base, data)
        print("[*](unicorn) initializing cpu registers")
        self.mu.reg_write(UC_X86_REG_RAX,self.dp.regs.rax)
        self.mu.reg_write(UC_X86_REG_RCX,self.dp.regs.rcx)
        self.mu.reg_write(UC_X86_REG_RDX,self.dp.regs.rdx)
        self.mu.reg_write(UC_X86_REG_RBX,self.dp.regs.rbx)
        self.mu.reg_write(UC_X86_REG_RSP,self.dp.regs.rsp)
        self.mu.reg_write(UC_X86_REG_RBP,self.dp.regs.rbp)
        self.mu.reg_write(UC_X86_REG_RSI,self.dp.regs.rsi)
        self.mu.reg_write(UC_X86_REG_RDI,self.dp.regs.rdi)
        self.mu.reg_write(UC_X86_REG_R8,self.dp.regs.r8)
        self.mu.reg_write(UC_X86_REG_R9,self.dp.regs.r9)
        self.mu.reg_write(UC_X86_REG_R10,self.dp.regs.r10)
        self.mu.reg_write(UC_X86_REG_R11,self.dp.regs.r11)
        self.mu.reg_write(UC_X86_REG_R12,self.dp.regs.r12)
        self.mu.reg_write(UC_X86_REG_R13,self.dp.regs.r13)
        self.mu.reg_write(UC_X86_REG_R14,self.dp.regs.r14)
        self.mu.reg_write(UC_X86_REG_R15,self.dp.regs.r15)
        print("[+] complete init execution context")

vmp = VmpAnalyzerX64('hdl.dmp',{'start':0,'end':0xFFFFFFFFFFFFFFFF})
vmp.analyze()
# load_minidump -> search vmenter -> processing vm handles
