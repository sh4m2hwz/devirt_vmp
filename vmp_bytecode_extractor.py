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

def get_reg(ctx,reg):
    if reg == ctx.registers.rax:
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
    elif reg == ctx.registers.r15:
        return UC_X86_REG_R15
    elif reg == ctx.registers.rip:
        return UC_X86_REG_RIP
    else:
        return UC_X86_REG_INVALID

is_stop = False
is_call = False
def hook_insn(mu, address, size, hook_ctx):
    global is_stop
    global is_call
    if is_stop:
        is_stop = False
        mu.emu_stop()
        return
    ctx = hook_ctx['ctx']
    vmp_range = hook_ctx['vmp_range']
    bb = hook_ctx['bb']
    insn = Instruction(address,bytes(mu.mem_read(address,size+1)))
    ctx.disassembly(insn)
    if is_call:
        is_call = False
        if not check_in_region(vmp_range, address):
            print("[+] found external call:",insn)
            mu.emu_stop()
            exit(-1)
    if insn.getType() == OPCODE.X86.CALL:
        is_call = True
        #bb.add(insn)
    elif insn.getType() == OPCODE.X86.JMP \
    and insn.getOperands()[0].getType() == OPERAND.REG:
        bb.add(insn)
        is_stop = True
    elif insn.getType() == OPCODE.X86.RET:
        bb.add(insn)
        is_stop = True
    elif not insn.isControlFlow():
        bb.add(insn)

bytecode_vars = {}

def replace_bytecode_vars(e_s,expr):
    global bytecode_vars
    if expr in bytecode_vars:
        return bytecode_vars[expr]
    return expr

class VmpAnalyzerX64:
    def __init__(self, filenamedump,vmp_segment_range={"start":0,"end":0}):
        self.filenamedump = filenamedump
        self.dp = Dumpulator(self.filenamedump,quiet=True)
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
        self.load_binary(self.dp)
        self.__init_triton_dse()
        self.symbolizeRegisters()
        self.hook_ctx = {"vmp_range":self.vmp_segment_range,"ctx":self.ctx,'bb':None}
        self.mu.hook_add(UC_HOOK_CODE,hook_insn,self.hook_ctx)
        self.entry_point = self.dp.regs.rip
        self.fetch_values = {
            "VPUSHI64" : [],
            "VADDQ": [],
            "VANDNQ": [],
            "VORNQ": [],
            "VPUSHRQ": [],
            "VPOPQ": [],
            "VMSWITCH": [],
            "VMENTER": [],
            "VLOADQ": [],
            "VLOADQSTACK": [],
            "VLoadToVSP": [],
            "VPUSHVSP": []
        }
    def get_dp(self):
        return self.dp
    def get_mu(self):
        return self.mu
    def get_ctx(self):
        return self.ctx
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
    def rebuildBasicBlock(self,bb):
        pc = bb.getFirstAddress()
        rebuilded = BasicBlock()
        for e in bb.getInstructions():
            rebuilded.add(Instruction(pc, e.getOpcode()))
            pc+=e.getSize()
        return rebuilded
    #
    def analyze_vmenter(self):
        attach_address = 0  
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
                        attach_address = insn.getAddress()
                        print("[+] found MOV reg, [RSP+90], calc VIP:",insn)
                        self.VIP=self.ctx.getParentRegister(op1)
                        print("[*] mapped reg VIP is:",ops[0].getName())
            if insn.getType() == OPCODE.X86.MOV:
                ops = insn.getOperands()
                if ops[0].getType() == OPERAND.REG \
                and ops[1].getType() == OPERAND.REG \
                and ops[1] == self.ctx.registers.rsp:
                    print("[+] found MOV contains VSP:",insn)
                    self.VSP=self.ctx.getParentRegister(ops[0])
                    self.VREGISTERS=self.ctx.getParentRegister(ops[1])
                    print("[*] mapped reg VSP is:",ops[0].getName())
                    print("[*] mapped reg VREGISTERS is:",ops[1].getName())
            elif insn.getType() == OPCODE.X86.POP:
                ops = insn.getOperands()
                if ops[0].getType() == OPERAND.REG:
                    print("[+] found POP contains RKEY:",insn)
                    self.RKEY = self.ctx.getParentRegister(ops[0])
            elif insn.getType() == OPCODE.X86.PUSH and insn.getOperands()[0].getType() == OPERAND.REG:
                self.reg_pco = self.ctx.getParentRegister(insn.getOperands()[0])
            elif insn.getType() == OPCODE.X86.RET:
                print("[+] found PUSH REG; RET pattern")
                print("[+] reg pco (Path Constraint) is:",self.reg_pco.getName())
                break
            elif insn.getType() == OPCODE.X86.JMP \
            and insn.getOperands()[0].getType() == OPERAND.REG:
                print("[+] found JMP REG pattern")
                self.reg_pco = self.ctx.getParentRegister(insn.getOperands()[0])
                print("[+] reg pco (Path Contraint) is:",self.reg_pco.getName())
                break
        if not self.reg_pco or not self.RKEY or not self.VSP \
        or not self.VREGISTERS or not self.VIP:
            print("[-] this devirtualizer support vmprotect 3.6.x, different version")
            exit(1)
        self.fetch_values['VMENTER'].append(attach_address)
        print("[+] complete finishing analyzing vmenter handler")
    #
    def analyze_vmswitch(self,bb):
        is_mov_reg_vsp = False
        is_mov_reg_mem_vsp = False
        reg_VIP_inter = None
        is_mov_new_vip = False
        attach_address = 0
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG \
            and ops[1] == self.VSP:
                is_mov_reg_vsp = True
                attach_address = insn.getAddress()
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
            self.fetch_values['VMSWITCH'].append(attach_address)
            print("[+] found vmswitch handler at address:",hex(bb.getFirstAddress()))
            return True
        return False
    #
    def analyze_vmexit(self,bb):
        pass
    #
    def find_VPUSHRQ(self,bb):
        index_reg = None
        val_reg = None
        bb_semantic = BasicBlock()
        is_mov_idx_reg_vip = False
        is_add_vip_1 = False
        is_mov_val_vregs_idx = False
        is_sub_vsp_8 = False
        is_mov_vsp_val = False
        attach_address = 0x0
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOVZX \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and self.VIP.getId() in [e[0].getId() for e in insn.getReadRegisters()] \
            and not is_mov_idx_reg_vip:
                is_mov_idx_reg_vip = True
                index_reg = self.ctx.getParentRegister(ops[0])
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.ADD \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x01 \
            and ops[0] == self.VIP \
            and not is_add_vip_1:
                is_add_vip_1 = True
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.SUB \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x01 \
            and ops[0] == self.VIP \
            and not is_add_vip_1:
                is_add_vip_1 = True
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == self.VREGISTERS \
            and ops[1].getIndexRegister() == index_reg \
            and not is_mov_val_vregs_idx:
                val_reg = ops[0]
                is_mov_val_vregs_idx = True
                attach_address = insn.getAddress()
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.SUB \
            and ops[0].getType() == OPERAND.REG \
            and ops[0] == self.VSP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x08 \
            and not is_sub_vsp_8:
                is_sub_vsp_8 = True
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == val_reg and not is_mov_vsp_val:
                is_mov_vsp_val = True
                bb_semantic.add(insn)
        if is_mov_vsp_val and is_sub_vsp_8 \
        and is_mov_val_vregs_idx and is_add_vip_1 \
        and is_mov_idx_reg_vip:
            self.fetch_values['VPUSHRQ'].append(attach_address)
            #print("[+] found VPUSHRQ handler")
            return True
        return False
    #
    def find_VPUSHI64(self,bb):
        constant_reg = None
        is_mov_const_reg_VIP = False
        is_mov_VSP_const_reg = False
        is_add_VIP_8 = False
        is_sub_VSP_8 = False
        attach_address = 0
        bb_semantic = BasicBlock()
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and self.VIP.getId() in [e[0].getId() for e in insn.getReadRegisters()] \
            and not is_mov_const_reg_VIP:
                is_mov_const_reg_VIP = True
                constant_reg = ops[0]
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == constant_reg \
            and not is_mov_VSP_const_reg:
                is_mov_VSP_const_reg = True
                attach_address = insn.getAddress()
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x08 and not is_add_VIP_8:
                is_add_VIP_8 = True
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x08 and not is_add_VIP_8:
                is_add_VIP_8 = True
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.SUB \
            and ops[0] == self.VSP and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x08 and not is_sub_VSP_8:
                is_sub_VSP_8 = True
                bb_semantic.add(insn)
        if is_sub_VSP_8 and is_add_VIP_8 and is_mov_const_reg_VIP and is_mov_VSP_const_reg:
            self.fetch_values['VPUSHI64'].append(attach_address)
            #print("[+] found VPUSHI64 handler")
            return True
        return False
    #
    def find_VPOPQ(self,bb):
        taint_reg = None
        index_reg = None
        is_mov_reg_vsp_mem = False
        is_add_vsp_8 = False
        is_mov_vregs_reg = False
        is_add_vip_1 = False
        is_mov_idx_vip = False
        attach_address = 0
        bb_semantic = BasicBlock()
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV:
                if ops[0].getType() == OPERAND.REG \
                and ops[1].getType() == OPERAND.MEM \
                and self.VSP.getId() in [e[0].getId() for e in insn.getReadRegisters()] \
                and not is_mov_reg_vsp_mem:
                    taint_reg = ops[0]
                    is_mov_reg_vsp_mem = True
                    bb_semantic.add(insn)
                elif ops[0].getType() == OPERAND.MEM \
                and ops[1].getType() == OPERAND.REG:
                    base = ops[0].getBaseRegister()
                    index = ops[0].getIndexRegister()
                    if base and index \
                    and base == self.VREGISTERS \
                    and index == index_reg \
                    and ops[1] == taint_reg \
                    and not is_mov_vregs_reg:
                        is_mov_vregs_reg = True
                        attach_address = insn.getAddress()
                        bb_semantic.add(insn)
            if opcode == OPCODE.X86.MOVZX and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and self.VIP.getId() in [e[0].getId() for e in insn.getReadRegisters()] \
            and not is_mov_idx_vip:
                index_reg = self.ctx.getParentRegister(ops[0])
                is_mov_idx_vip = True
                bb_semantic.add(insn)
            elif opcode == OPCODE.X86.ADD \
            and ops[0] == self.VSP and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x08 and not is_add_vsp_8:
                is_add_vsp_8 = True
                bb_semantic.add(insn)
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x01 and not is_add_vip_1:
                is_add_vip_1 = True
                bb_semantic.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x01 and not is_add_vip_1:
                is_add_vip_1 = True
                bb_semantic.add(insn)
        if is_mov_idx_vip and is_mov_vregs_reg and is_add_vsp_8 \
        and is_mov_reg_vsp_mem and is_add_vip_1:
            self.fetch_values['VPOPQ'].append(attach_address)
            #print("[+] found VPOPQ handler")
            return True
        return False
    #
    def find_VADDQ(self,bb):
        bb_semantics = BasicBlock()
        reg_a = None
        reg_b = None
        is_mov_a = False
        is_mov_b = False
        is_add_a_b = False
        is_mov_res = False
        is_pushfq = False
        is_pop_VSP = False
        attach_address = 0
        is_add_VIP_4 = False
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP and \
            ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_VIP_4 = True
                attach_address = insn.getAddress()
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP and \
            ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_VIP_4 = True
                attach_address = insn.getAddress()
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == self.VSP:
                disp = ops[1].getDisplacement()
                if not disp or disp.getValue() == 0:
                    is_mov_a = True
                    reg_a = ops[0]
                else:
                    is_mov_b = True
                    reg_b = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[0].getDisplacement() \
            and ops[0].getDisplacement().getValue() == 0x08 \
            and (ops[1] == reg_a or ops[1] == reg_b):
                is_mov_res = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.ADD \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG \
            and reg_a in [e[0] for e in insn.getReadRegisters()] \
            and reg_b in [e[0] for e in insn.getReadRegisters()]:
                is_add_a_b = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.PUSHFQ:
                is_pushfq = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.POP \
            and ops[0].getType() == OPERAND.MEM \
            and ops[0].getBaseRegister() == self.VSP:
                is_pop_VSP = True
                bb_semantics.add(insn)
        if is_pop_VSP and is_pushfq and is_add_a_b and is_mov_res \
        and is_mov_b and is_mov_a:
            self.fetch_values['VADDQ'].append(attach_address)
            #print("[+] found VADDQ handler")
            return True
        return False
    #
    def find_VLOADQ(self,bb):
        reg_ptr = None
        reg_val = None
        is_mov_reg_ptr_vsp = False
        is_mov_val_reg_ptr = False
        is_mov_vsp_val = False
        is_add_vip_4 = False
        attach_address = 0
        bb_semantics = BasicBlock()
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == self.VSP:
                is_mov_reg_ptr_vsp = True
                reg_ptr = ops[0]
                bb_semantics.add(insn)
                attach_address = insn.getAddress()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == reg_ptr \
            and ops[1].getSegmentRegister() != self.ctx.registers.ss:
                is_mov_val_reg_ptr = True
                reg_val = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == reg_val:
                is_mov_vsp_val = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)       
        if is_add_vip_4 and is_mov_vsp_val \
        and is_mov_val_reg_ptr and is_mov_reg_ptr_vsp:
            self.fetch_values['VLOADQ'].append(attach_address)
            #print("[+] found VLOADQ handler")
            return True
        return False
    #
    def find_VPUSHVSP(self,bb):
        vsp_val_reg = None
        is_mov_vsp_val_reg = False
        is_sub_vsp_8 = False
        is_mov_vsp_vsp_val_reg = False
        is_add_vip_4 = False
        attach_address = 0
        bb_semantics = BasicBlock()
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG \
            and ops[1] == self.VSP:
                vsp_val_reg = ops[0]
                is_mov_vsp_val_reg = True
                bb_semantics.add(insn)
                attach_address = insn.getAddress()
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VSP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x08:
                is_sub_vsp_8 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == vsp_val_reg:
                is_mov_vsp_vsp_val_reg = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
        if is_add_vip_4 and is_sub_vsp_8 and is_mov_vsp_vsp_val_reg \
        and is_mov_vsp_val_reg:
            #print("[+] found VPUSHVSP handler")
            self.fetch_values['VPUSHVSP'].append(attach_address)
            return True
        return False
    #
    def find_VANDNQ(self,bb):
        is_mov_a = False
        is_mov_b = False
        reg_a = None
        reg_b = None
        reg_res = None
        is_not_a = False
        is_not_b = False
        is_or_a_b = False
        is_mov_vsp_res = False
        is_pushfq = False
        is_pop_rflags = False
        is_add_vip_4 = False
        attach_address = 0
        bb_semantics = BasicBlock()
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == self.VSP:
                disp = ops[1].getDisplacement()
                if (not disp) or (disp.getValue() == 0):
                    is_mov_a = True
                    reg_a = ops[0]
                else:
                    is_mov_b = True
                    reg_b = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.NOT \
            and ops[0].getType() == OPERAND.REG \
            and (ops[0] == reg_a or ops[0] == reg_b):
                if ops[0] == reg_a:
                    is_not_a = True
                else:
                    is_not_b = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.OR \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG \
            and reg_a in [e[0] for e in insn.getReadRegisters()] \
            and reg_b in [e[0] for e in insn.getReadRegisters()]:
                is_or_a_b = True
                reg_res = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == reg_res:
                is_mov_vsp_res = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.PUSHFQ:
                is_pushfq = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.POP \
            and ops[0].getType() == OPERAND.MEM \
            and ops[0].getBaseRegister() == self.VSP:
                is_pop_rflags = True
                bb_semantics.add(insn)
                attach_address = insn.getAddress()
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
        if is_add_vip_4 and is_pop_rflags and is_pushfq \
        and is_mov_vsp_res and is_or_a_b and is_not_a and is_not_b \
        and is_mov_a and is_mov_b:
            self.fetch_values['VANDNQ'].append(attach_address)
            #print("[+] found VANDNQ handler")
            return True
        return False
    #
    def find_VORNQ(self,bb):
        is_mov_a = False
        is_mov_b = False
        reg_a = None
        reg_b = None
        reg_res = None
        is_not_a = False
        is_not_b = False
        is_and_a_b = False
        is_mov_vsp_res = False
        is_pushfq = False
        is_pop_rflags = False
        is_add_vip_4 = False
        attach_address = 0
        bb_semantics = BasicBlock()
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == self.VSP:
                disp = ops[1].getDisplacement()
                if (not disp) or (disp.getValue() == 0):
                    is_mov_a = True
                    reg_a = ops[0]
                else:
                    is_mov_b = True
                    reg_b = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.NOT \
            and ops[0].getType() == OPERAND.REG \
            and (ops[0] == reg_a or ops[0] == reg_b):
                if ops[0] == reg_a:
                    is_not_a = True
                else:
                    is_not_b = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.AND \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.REG \
            and reg_a in [e[0] for e in insn.getReadRegisters()] \
            and reg_b in [e[0] for e in insn.getReadRegisters()]:
                is_and_a_b = True
                reg_res = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == reg_res:
                is_mov_vsp_res = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.PUSHFQ:
                is_pushfq = True
                bb_semantics.add(insn)
                attach_address = insn.getAddress()
            if opcode == OPCODE.X86.POP \
            and ops[0].getType() == OPERAND.MEM \
            and ops[0].getBaseRegister() == self.VSP:
                is_pop_rflags = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
        if is_add_vip_4 and is_pop_rflags and is_pushfq \
        and is_mov_vsp_res and is_and_a_b and is_not_a and is_not_b \
        and is_mov_a and is_mov_b:
            self.fetch_values['VORNQ'].append(attach_address)
            #print("[+] found VORNQ handler")
            return True
        return False
    #
    def find_VLoadToVSP(self,bb):
        is_mov_vsp_mem_vsp = False
        is_add_vip_4 = False
        bb_semantics = BasicBlock()
        attach_address = 0
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and self.VSP in [e[0] for e in insn.getReadRegisters()] \
            and self.VSP in [e[0] for e in insn.getWrittenRegisters()]:
                is_mov_vsp_mem_vsp = True
                bb_semantics.add(insn)
                attach_address = insn.getAddress()
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
        if is_add_vip_4 and is_mov_vsp_mem_vsp:
            self.fetch_values['VLoadToVSP'].append(attach_address)
            #print("[+] found VLoadToVSP handler")
            return True
        return False
    #
    def find_VLOADQSTACK(self,bb):
        reg_ptr = None
        reg_val = None
        is_mov_reg_ptr_vsp = False
        is_mov_val_reg_ptr = False
        is_mov_vsp_val = False
        is_add_vip_4 = False
        bb_semantics = BasicBlock()
        attach_address = 0
        for insn in bb.getInstructions():
            opcode = insn.getType()
            ops = insn.getOperands()
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == self.VSP:
                is_mov_reg_ptr_vsp = True
                reg_ptr = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.REG \
            and ops[1].getType() == OPERAND.MEM \
            and ops[1].getBaseRegister() == reg_ptr \
            and ops[1].getSegmentRegister() == self.ctx.registers.ss:
                is_mov_val_reg_ptr = True
                reg_val = ops[0]
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.MOV \
            and ops[0].getType() == OPERAND.MEM \
            and ops[1].getType() == OPERAND.REG \
            and ops[0].getBaseRegister() == self.VSP \
            and ops[1] == reg_val:
                is_mov_vsp_val = True
                attach_address = insn.getAddress()
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.ADD \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)
            if opcode == OPCODE.X86.SUB \
            and ops[0] == self.VIP \
            and ops[1].getType() == OPERAND.IMM \
            and ops[1].getValue() == 0x04:
                is_add_vip_4 = True
                bb_semantics.add(insn)       
        if is_add_vip_4 and is_mov_vsp_val \
        and is_mov_val_reg_ptr and is_mov_reg_ptr_vsp:
            self.fetch_values['VLOADQSTACK'].append(attach_address)
            #print("[+] found VLOADQSTACK handler")
            return True
        return False
    #
    def analyze_vmops(self,bb):
        analyzers = [
            self.find_VPUSHRQ,
            self.find_VPUSHI64,
            self.find_VPOPQ,
            self.find_VADDQ,
            self.find_VLOADQ,
            self.find_VPUSHVSP,
            self.find_VANDNQ,
            self.find_VORNQ,
            self.find_VLOADQSTACK,
            self.find_VLoadToVSP
        ]
        for analyzer in analyzers:
            if analyzer(bb):
                return True
        return False
    #
    def analyze_vmhandlers(self,bb):
        #print("[*] analyze vmp handler\n")
        if self.analyze_vmswitch(bb):
            return True
        if self.analyze_vmexit(bb):
            return True
        if not self.analyze_vmops(bb):
            print("[-] not found vmp handler, please append vmp template handler")
            exit(1)
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
                    self.symbolizeRegistersBytecodeCtx()
                    print("[+] start analyzing vmp handlers")
            else:
                is_vmswitch_or_vmexit = self.analyze_vmhandlers(bb)
                if is_vmswitch_or_vmexit:
                    break
            bb_entry = self.getNextEntryBasicBlock()
        return self.fetch_values
    #
    def symbolizeRegisters(self):
        print("[*] symbolizing all registers")
        for reg in ['rax','rcx','rdx','rbx','rsp','rbp','rsi','rdi','r8','r9','r10','r11','r12','r13','r14','r15','rip','eflags']:
            self.ctx.symbolizeRegister(self.ctx.getRegister(reg),reg)
    def symbolizeRegistersBytecodeCtx(self):
        self.ctx_bytecode.symbolizeRegister(self.VIP)
        self.ctx_bytecode.symbolizeRegister(self.VSP)
        self.ctx_bytecode.symbolizeRegister(self.VREGISTERS)
    #
    def init_triton_dse_regs(self,ctx):
        ctx.setConcreteRegisterValue(ctx.registers.rax,self.dp.regs.rax)
        ctx.setConcreteRegisterValue(ctx.registers.rcx,self.dp.regs.rcx)
        ctx.setConcreteRegisterValue(ctx.registers.rdx,self.dp.regs.rdx)
        ctx.setConcreteRegisterValue(ctx.registers.rbx,self.dp.regs.rbx)
        ctx.setConcreteRegisterValue(ctx.registers.rsp,self.dp.regs.rsp)
        ctx.setConcreteRegisterValue(ctx.registers.rbp,self.dp.regs.rbp)
        ctx.setConcreteRegisterValue(ctx.registers.rsi,self.dp.regs.rsi)
        ctx.setConcreteRegisterValue(ctx.registers.rdi,self.dp.regs.rdi)
        ctx.setConcreteRegisterValue(ctx.registers.r8,self.dp.regs.r8)
        ctx.setConcreteRegisterValue(ctx.registers.r9,self.dp.regs.r9)
        ctx.setConcreteRegisterValue(ctx.registers.r10,self.dp.regs.r10)
        ctx.setConcreteRegisterValue(ctx.registers.r11,self.dp.regs.r11)
        ctx.setConcreteRegisterValue(ctx.registers.r12,self.dp.regs.r12)
        ctx.setConcreteRegisterValue(ctx.registers.r13,self.dp.regs.r13)
        ctx.setConcreteRegisterValue(ctx.registers.r14,self.dp.regs.r14)
        ctx.setConcreteRegisterValue(ctx.registers.r15,self.dp.regs.r15)
    #
    def __init_triton_dse(self):
        print("[*](triton dse) init ctx")
        self.ctx = TritonContext(ARCH.X86_64)
        self.ctx_bytecode = TritonContext(ARCH.X86_64)
        self.ctx.setAstRepresentationMode(AST_REPRESENTATION.PCODE)
        self.ctx.setAstRepresentationMode(AST_REPRESENTATION.PCODE)
        self.ctx_bytecode.setMode(MODE.ALIGNED_MEMORY, True)
        self.ctx_bytecode.setMode(MODE.CONSTANT_FOLDING, True)
        self.ctx_bytecode.setMode(MODE.AST_OPTIMIZATIONS, True)
        self.ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self.ctx.setMode(MODE.CONSTANT_FOLDING, True)
        self.ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        self.init_triton_dse_regs(self.ctx)
        self.init_triton_dse_regs(self.ctx_bytecode)
        print("[*](triton dse) mapping and writing datas into regions")
        for region in self.mu.mem_regions():
            print(f"[*](triton dse) mapping and writing data into region: va_range={hex(region[0])}-{hex(region[1])}")
            data = self.mu.mem_read(region[0],region[1]-region[0])
            self.ctx.setConcreteMemoryAreaValue(region[0], data)
            self.ctx_bytecode.setConcreteMemoryAreaValue(region[0], data)
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
    def load_binary(self,dp):
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

def hook_extractor_bytecode(mu, address, size, vmp_extractor):
    if vmp_extractor.is_vmp_handler(address):
        vmp_extractor.gen(mu,address)

VPUSHI64 =  0
VPUSHRQ  =  1
VPUSHVSP =  2
VLoadToVSP= 3
VLOADQSTACK=4
VLOADQ    = 5
VPOPQ     = 6
VADDQ     = 7
VANDNQ    = 8
VORNQ     = 9
VMENTER  = 10
VMSWITCH = 11

class VmpExtractorX64(VmpAnalyzerX64):
    def __init__(self,minidumpfile,vmp_segment_range):
        super().__init__(minidumpfile,vmp_segment_range)
        self.vmp_handlers_attach_list = None
        self.vmp_bytecode = bytes()
    #
    def extract_bytecode(self):
        self.vmp_handlers_attach_list = super().analyze()
        print("[*] start extracting bytecode")
        super().load_binary(super().get_dp())
        super().get_mu().hook_add(UC_HOOK_CODE,hook_extractor_bytecode,self)
        super().get_mu().emu_start(super().get_dp().regs.rip,-1)
        print("[+] complete extract bytecode!")
        return self.vmp_bytecode
    #
    def is_vmp_handler(self,address_handler):
        for l in self.vmp_handlers_attach_list.values():
            if address_handler in l:
                return True
        return False   
    #
    def check_handler(self,name,address_handler):
        if address_handler not in self.vmp_handlers_attach_list[name]:
            return False
        l = self.vmp_handlers_attach_list[name]
        for i in range(len(l)):
            if l[i] == address_handler:
                l.pop(i)
                break
        return True
    #
    def gen_imm64(self,imm64):
        self.vmp_bytecode+=int.to_bytes(imm64,8,byteorder='little')
    def gen_opcode(self,opcode):
        self.vmp_bytecode+=int.to_bytes(opcode,1,byteorder='little')
    def gen_regop(self,reg_idx):
        self.vmp_bytecode+=int.to_bytes(reg_idx,1,byteorder='little')
    #
    def gen_VPUSHI64(self,mu,address_handler):
        if not self.check_handler("VPUSHI64",address_handler):
            return
        insn = Instruction(address_handler,bytes(mu.mem_read(address_handler,16)))
        super().get_ctx().disassembly(insn)
        reg_id = get_reg(super().get_ctx(),insn.getOperands()[1])
        i64 = mu.reg_read(reg_id)
        self.gen_opcode(VPUSHI64)
        self.gen_imm64(i64)
    #
    def gen_VPUSHRQ(self,mu,address_handler):
        if not self.check_handler("VPUSHRQ",address_handler):
            return
        insn = Instruction(address_handler,bytes(mu.mem_read(address_handler,16)))
        super().get_ctx().disassembly(insn)
        reg_id = get_reg(super().get_ctx(),insn.getOperands()[1].getIndexRegister())
        idx = mu.reg_read(reg_id)
        self.gen_opcode(VPUSHRQ)
        self.gen_regop(idx)
    #
    def gen_VPOPQ(self,mu,address_handler):
        if not self.check_handler("VPOPQ",address_handler):
            return
        insn = Instruction(address_handler,bytes(mu.mem_read(address_handler,16)))
        super().get_ctx().disassembly(insn)
        reg_id = get_reg(super().get_ctx(),insn.getOperands()[0].getIndexRegister())
        idx = mu.reg_read(reg_id)
        self.gen_opcode(VPOPQ)
        self.gen_regop(idx)
    #
    def gen_VADDQ(self,mu,address_handler):
        if not self.check_handler("VADDQ",address_handler):
            return
        self.gen_opcode(VADDQ)
    #
    def gen_VMENTER(self,mu,address_handler):
        if not self.check_handler("VMENTER",address_handler):
            return
        self.gen_opcode(VMENTER)
    #
    def gen_VMSWITCH(self,mu,address_handler):
        if not self.check_handler("VMSWITCH",address_handler):
            return
        self.gen_opcode(VMSWITCH)
        mu.emu_stop()
    #
    def gen_VANDNQ(self,mu,address_handler):
        if not self.check_handler("VANDNQ",address_handler):
            return
        self.gen_opcode(VANDNQ)
    #
    def gen_VORNQ(self,mu,address_handler):
        if not self.check_handler("VORNQ",address_handler):
            return
        self.gen_opcode(VORNQ)
    #
    def gen_VPUSHVSP(self,mu,address_handler):
        if not self.check_handler("VPUSHVSP",address_handler):
            return
        self.gen_opcode(VPUSHVSP)
    #
    def gen_VLoadToVSP(self,mu,address_handler):
        if not self.check_handler("VLoadToVSP",address_handler):
            return
        self.gen_opcode(VLoadToVSP)
    #
    def gen_VLOADQ(self,mu,address_handler):
        if not self.check_handler("VLOADQ",address_handler):
            return
        self.gen_opcode(VLOADQ)
    #
    def gen_VLOADQSTACK(self,mu,address_handler):
        if not self.check_handler("VLOADQSTACK",address_handler):
            return
        self.gen_opcode(VLOADQSTACK)
    #
    def gen(self,mu,address_handler):
        handlers = [
            self.gen_VADDQ,
            self.gen_VANDNQ,
            self.gen_VLOADQ,
            self.gen_VLOADQSTACK,
            self.gen_VLoadToVSP,
            self.gen_VMENTER,
            self.gen_VMSWITCH,
            self.gen_VORNQ,
            self.gen_VPOPQ,
            self.gen_VPUSHI64,
            self.gen_VPUSHRQ
        ]
        for hdl in handlers:
            if hdl(mu,address_handler):
               return

vmp = VmpExtractorX64('hdl.dmp',{'start':0,'end':0xFFFFFFFFFFFFFFFF})
vmp_bytecode = vmp.extract_bytecode()
