local torch = require 'torch'
local argcheck = require 'argcheck'

local M = {}

local ffi = require 'ffi'
ffi.load("/home/bamos/src/gurobi650/linux64/lib/libgurobi65.so", true)
ffi.cdef [[
typedef void GRBenv;
typedef void GRBmodel;

GRBenv* GT_loadenv(const char *logfilename, int outputFlag);
GRBmodel* GT_newmodel(GRBenv *env, const char *name, THDoubleTensor *obj,
                                  THDoubleTensor *lb, THDoubleTensor *ub);

void GT_addconstr(GRBmodel *model, int nnz, THIntTensor *cind, THDoubleTensor *cval,
               const char *sense, double rhs);

void GT_addqpterms(GRBmodel *model, int numqnz, THIntTensor *qrow, THIntTensor *qcol,
                   THDoubleTensor *qval);
void GT_delq(GRBmodel *model);

int GT_solve(THDoubleTensor *rx, GRBmodel *model);
void GT_solvePar(THDoubleTensor *rx, THIntTensor *status, GRBmodel **models,
                 int nVars, int nModels);

void GT_setdblattrlist(GRBmodel *model, const char *name, int len, THIntTensor *ind,
                       THDoubleTensor *values);
int GT_getintattr(GRBmodel *model, const char *name);

int GT_free(GRBenv *env, GRBmodel *model);
]]

local clib = ffi.load(package.searchpath('libgurobi', package.cpath))

local loadenvCheck = argcheck{
   pack=true,
   {name='logfilename', type='string'},
   {name='outputFlag', type='number', opt=true, default=0}
}
function M.loadenv(...)
   local args = loadenvCheck(...)
   return clib.GT_loadenv(args.logfilename, args.outputFlag)
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

   return clib.GT_newmodel(args.env, args.name, obj_:cdata(), lb_:cdata(), ub_)
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
   clib.GT_addconstr(model, nnz, cind:cdata(), cval:cdata(), sense, rhs)
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

local addqptermsCheck = argcheck{
   pack=true,
   {name='model', type='cdata'},
   {name='Q', type='torch.*Tensor'},
}
function M.addqpterms(...)
   local args = addqptermsCheck(...)
   local model = args.model
   local Q = args.Q

   local nz = Q:nonzero():int():split(1, 2)
   local qrow = nz[1]:clone():view(-1)
   local qcol = nz[2]:clone():view(-1)
   local numqnz = qrow:nElement()
   local qval = torch.DoubleTensor(numqnz)
   for i = 1, numqnz do
      qval[i] = Q[qrow[i]][qcol[i]]
   end
   qrow:csub(1.0)
   qcol:csub(1.0)

   clib.GT_addqpterms(model, numqnz, qrow:cdata(), qcol:cdata(), qval:cdata())
end

function M.delq(model)
   clib.GT_delq(model)
end

function M.solve(model)
   local nvars = clib.GT_getintattr(model, "NumVars")
   local rx = torch.DoubleTensor(nvars)
   local status = clib.GT_solve(rx:cdata(), model)
   return status, rx
end

function M.solvePar(models)
   local nvars = clib.GT_getintattr(models[1], "NumVars")
   local nModels = table.getn(models)
   local rx = torch.DoubleTensor(nModels, nvars)
   print(nModels)
   local modelsFfi = ffi.new("GRBmodel *[" .. nModels .. "]", models)
   local status = torch.IntTensor(nModels)
   clib.GT_solvePar(rx:cdata(), status:cdata(), modelsFfi, nvars, nModels)
   return status, rx
end

function M.updateObj(model, obj)
   M.setdblattrlist(model, "Obj", obj)
end

function M.setdblattrlist(model, name, x)
   local ind = x:nonzero():int()-1
   local val = x[torch.ne(x, 0.0)]
   local nnz = ind:nElement()
   clib.GT_setdblattrlist(model, name, nnz, ind:cdata(), val:cdata())
end

function M.free(env, model)
   clib.GT_free(env, model)
end

return M
