from miasm.expression.expression import ExprId
from miasm.core.cpu import gen_regs

exception_flags = ExprId("exception_flags", 32)
exception_flags_init = ExprId("exception_flags_init", 32)

gpr_names = ["R%d" % r for r in range(255)]  # register names
gpr_exprs, gpr_inits, gpr_infos = gen_regs(gpr_names, globals(),sz=64)  # sz=32 bits (default)
# PC SP QP
csr_names = ["VIP", "VSP"]
csr_exprs, csr_inits, csr_infos = gen_regs(csr_names, globals(),sz=64)

PC = csr_exprs[0]
SP = csr_exprs[1]

PC_init = csr_inits[0]
SP_init = csr_inits[1]

# Set registers initial values
all_regs_ids = gpr_exprs + csr_exprs + [exception_flags]
all_regs_ids_init = gpr_inits + csr_inits + [exception_flags_init]
all_regs_ids_no_alias = all_regs_ids[:]
all_regs_ids_byname = dict([(x.name, x) for x in all_regs_ids])

regs_init = dict()  # mandatory name
for i, r in enumerate(all_regs_ids):
  regs_init[r] = all_regs_ids_init[i]