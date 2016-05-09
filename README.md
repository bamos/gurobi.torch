# gurobi.torch • [ ![Build Status] [travis-image] ] [travis] [ ![License] [license-image] ] [license]

*Unofficial Gurobi Torch bindings.*

[travis-image]: https://travis-ci.org/bamos/gurobi.torch.png?branch=master
[travis]: http://travis-ci.org/bamos/gurobi.torch

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

---

+ **Warning:** This package is unstable, unfinished, and under development.
  Please see our [issue tracker](https://github.com/bamos/gurobi.torch/issues)
  for unresolved issues.
  Contact [Brandon Amos](http://bamos.github.io) with any questions
  or issues.
+ You may also be interested in a Torch
  [ECOS](https://github.com/embotech/ecos) wrapper:
  [bamos/ecos.torch](https://github.com/bamos/ecos.torch)

# Installation

1. Update paths to your Gurobi installation in `CMakeLists.txt` and `init.c`.
2. `luarocks make`

# Usage

The following solves the linear program

```
min  c'*x s.t. G*x <= h
```

## Linear Program

```lua
local gurobi = require 'gurobi'

local G = torch.Tensor{{-1, 1}, {-1, -1}, {0, -1}, {1, -2}}
local h = torch.Tensor{1.0, -2.0, 0.0, 4.0}
local c = torch.Tensor{2.0, 1.0}

local env = gurobi.loadenv("")
local model = gurobi.newmodel(env, "", c)
gurobi.addconstrs(model, G, 'LE', h)
local status, x = gurobi.solve(model)
print(x) -- Optimal x is [0.5, 1.5]
```

## Linear Program with Incrementally Added Constraints
```lua
local gurobi = require 'gurobi'

local c = torch.Tensor{2.0, 1.0}
local G = torch.Tensor{{-1, 1}, {-1, -1}, {0, -1}, {1, -2}}
local h = torch.Tensor{1.0, -2.0, 0.0, 4.0}

local env = gurobi.loadenv("")
local model = gurobi.newmodel(env, "", c)
gurobi.addconstr(model, G[1], 'LE', h[1])

local status, x = gurobi.solve(model)
print(x) -- Optimal at this point is [0, 0]

local I = {{2,4}}
gurobi.addconstrs(model, G[I], 'LE', h[I])
status, x = gurobi.solve(model)
print(x) -- Optimal at this point is [0.5, 1.5]
```

# Tests

After installing the library with `luarocks`, our tests in
[test.lua](https://github.com/bamos/ecos.torch/blob/master/test.lua)
can be run with `th test.lua`.

# Licensing

+ Gurobi is proprietary software.
+ The original code in this repository (the Gurobi bindings) is
  [Apache-licensed](https://github.com/bamos/ecos.torch/blob/master/LICENSE).
