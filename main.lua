require 'nn'
require 'nngraph'

unigrid = require 'unigrid'

cmd = torch.CmdLine()
cmd:text('Train a 1 dimensional grid lstm')
cmd:text('Options')
cmd:option('-size', 200,'rnn size')
cmd:option('-k', 15, 'size of input bit vector')
cmd:option('-n', 5, 'number of layers')
opt = cmd:parse(arg)

local x = nn.Identity()()

local h0 = nn.Linear(opt.k, opt.size)(x)
local m0 = nn.Linear(opt.k, opt.size)(x)
local h, m = unigrid.unigrid(opt.size, opt.n, h0, m0)


--local y = nn.CAddTable()({
  --  nn.Linear(1, opt.size)(hn), 
    --nn.Linear(1, opt.size)(mn)
--})

local model = nn.gModule({x}, {h, m})
--xx = torch.randn(opt.k)
--model:forward(xx)

graph.dot(model.fg, 'mlp', 'mymlp')
