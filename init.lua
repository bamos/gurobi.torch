local torch = require 'torch'
local argcheck = require 'argcheck'

local M = {}

local ffi = require 'ffi'
ffi.load("/home/bamos/src/gurobi650/linux64/lib/libgurobi65.so", true)
ffi.cdef [[
void* /* GRBenv* */ loadenv(const char *logfilename, int outputFlag);
void* /* GRBmodel* */ newmodel(void *env, const char *name, THDoubleTensor *obj,
                               THDoubleTensor *lb, THDoubleTensor *ub);
void addconstr(void *model, int nnz, THIntTensor *cind, THDoubleTensor *cval,
               const char *sense, double rhs);
int solve(THDoubleTensor *rx, void *model);

int getintattr(void *model, const char *name);
]]

local clib = ffi.load(package.searchpath('libgurobi', package.cpath))

local loadenvCheck = argcheck{
   pack=true,
   {name='logfilename', type='string'},
   {name='outputFlag', type='number', opt=true, default=0}
}
function M.loadenv(...)
   local args = loadenvCheck(...)
   return clib.loadenv(args.logfilename, args.outputFlag)
end

local newmodelCheck = argcheck{
   pack=true,
   {name='env', type='cdata'},
   {name='name', type='string'},
   {name='obj', type='torch.*Tensor'},
   {name='lb', type='torch.*Tensor', opt=true},
   {name='ub', type='torch.*Tensor', opt=true}
}
function M.newmodel(...)
   local args = newmodelCheck(...)
   local obj_ = args.obj:double()

   local lb_, ub_

   if args.lb then lb_ = args.lb
   else lb_ = torch.DoubleTensor():resizeAs(obj_):fill(-1e99) end

   if args.ub then ub_ = args.ub:cdata() end

   return clib.newmodel(args.env, args.name, obj_:cdata(), lb_:cdata(), ub_)
end

local addconstrCheck = argcheck{
   pack=true,
   {name='model', type='cdata'},
   {name='lhs', type='torch.*Tensor'},
   {name='sense', type='string'},
   {name='rhs', type='number'}
}
function M.addconstr(...)
   local args = addconstrCheck(...)
   local model = args.model
   local lhs = args.lhs
   local sense = args.sense
   local rhs = args.rhs

   local cind = lhs:nonzero():int()-1
   local cval = lhs[torch.ne(lhs, 0.0)]
   local nnz = cind:nElement()
   clib.addconstr(model, nnz, cind:cdata(), cval:cdata(), sense, rhs)
end

local addconstrsCheck = argcheck{
   pack=true,
   {name='model', type='cdata'},
   {name='lhs', type='torch.*Tensor'},
   {name='sense', type='string'},
   {name='rhs', type='torch.*Tensor'}
}
function M.addconstrs(...)
   local args = addconstrsCheck(...)
   local model = args.model
   local lhs = args.lhs
   local sense = args.sense
   local rhs = args.rhs

   local nConstr = lhs:size(1)
   for i = 1, nConstr do
      M.addconstr(model, lhs[i], sense, rhs[i])
   end
end

function M.solve(model)
   local nvars = clib.getintattr(model, "NumVars")
   local rx = torch.DoubleTensor(nvars)
   local status = clib.solve(rx:cdata(), model)
   return status, rx
end

return M
