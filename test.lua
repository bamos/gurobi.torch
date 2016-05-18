#!/usr/bin/env th

local torch = require 'torch'
torch.setdefaulttensortype('torch.DoubleTensor')

local gurobi = require 'gurobi'

local tester = torch.Tester()
local gurobiTest = torch.TestSuite()

local eps = 1e-5

function gurobiTest.SmallLP()
   local env = gurobi.loadenv("")

   local c = torch.Tensor{2.0, 1.0}
   local G = torch.Tensor{{-1, 1}, {-1, -1}, {0, -1}, {1, -2}}
   local h = torch.Tensor{1.0, -2.0, 0.0, 4.0}

   local model = gurobi.newmodel(env, "", c)
   gurobi.addconstrs(model, G, 'LE', h)
   local status, x = gurobi.solve(model)

   local optX = torch.Tensor{0.5, 1.5}
   tester:asserteq(status, 2, 'Non-optimal status: ' .. status)
   tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')

   gurobi.free(env, model)
end

function gurobiTest.SmallLP_Incremental()
   -- minimize y
   -- subject to y >= x
   --            y >= -x
   --            y >= x + 1
   local c = torch.Tensor{0.0, 1.0}
   local G = torch.Tensor{{1, -1}, {-1, -1}, {1, -1}}
   local h = torch.Tensor{0.0, 0.0, -1.0}

   local env = gurobi.loadenv("")
   local model = gurobi.newmodel(env, "", c)

   local I = {{1,2}}
   gurobi.addconstrs(model, G[I], 'LE', h[I])

   local status, x = gurobi.solve(model)
   local optX = torch.Tensor{0.0, 0.0}
   tester:asserteq(status, 2, 'Non-optimal status: ' .. status)
   tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')

   gurobi.addconstr(model, G[3], 'LE', h[3])
   status, x = gurobi.solve(model)
   optX = torch.Tensor{-0.5, 0.5}
   tester:asserteq(status, 2, 'Non-optimal status: ' .. status)
   tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')

   gurobi.free(env, model)
end

function gurobiTest.SmallQP()
   local env = gurobi.loadenv("")
   local c = torch.Tensor{2.0, 1.0}
   local model = gurobi.newmodel(env, "", c)

   local Q = torch.eye(2)
   gurobi.addqpterms(model, Q)

   local status, x = gurobi.solve(model)

   local optX = torch.Tensor{-1.0, -0.5}
   tester:asserteq(status, 2, 'Non-optimal status: ' .. status)
   tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')

   gurobi.free(env, model)
end

function gurobiTest.ChangeObj()
   local env = gurobi.loadenv("")
   local c = torch.Tensor{0.0, 1.0}
   local lb = torch.Tensor{0.0, 0.0}
   local ub = torch.Tensor{3.0, 3.0}
   local model = gurobi.newmodel(env, "", c, lb, ub)

   local A = torch.Tensor{1.0, -1.0}
   gurobi.addconstr(model, A, 'EQ', 0.0)

   local status, x = gurobi.solve(model)

   local optX = torch.Tensor{0.0, 0.0}
   tester:asserteq(status, 2, 'Non-optimal status: ' .. status)
   tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')


   local newC = torch.Tensor{0.0, -1.0}
   gurobi.updateObj(model, newC)

   status, x = gurobi.solve(model)

   optX = torch.Tensor{3.0, 3.0}
   tester:asserteq(status, 2, 'Non-optimal status: ' .. status)
   tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')

   gurobi.free(env, model)
end

function gurobiTest.par()
   local c = torch.Tensor{2.0, 1.0}
   local G = torch.Tensor{{-1, 1}, {-1, -1}, {0, -1}, {1, -2}}
   local h = torch.Tensor{1.0, -2.0, 0.0, 4.0}

   local env = gurobi.loadenv("")
   local model1 = gurobi.newmodel(env, "", c)
   gurobi.addconstrs(model1, G, 'LE', h)

   local model2 = gurobi.newmodel(env, "", c)
   gurobi.addconstrs(model2, G, 'LE', h)

   local status, xs = gurobi.solvePar({model1, model2})

   local optX = torch.Tensor{0.5, 1.5}
   for i = 1,2 do
      local status_i = status[i]
      local x = xs[i]
      tester:asserteq(status_i, 2, 'Non-optimal status: ' .. status_i)
      tester:assertTensorEq(x, optX, eps, 'Invalid optimal value.')
   end

   gurobi.free(nil, model1)
   gurobi.free(env, model2)
end

tester:add(gurobiTest)
tester:run()
