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

local h0 = nn.Linear(opt.size, opt.k)(x)
local m0 = nn.Linear(opt.size, opt.k)(x)
local hn, mn = unpack(unigrid.unigrid(opt.n, opt.size)({h0, m0}))


local y = nn.CAddTable()({
    nn.Linear(1, opt.size)(hn), 
    nn.Linear(1, opt.size)(mn)
})

local model = nn.gModule({x}, {hn, mn})
