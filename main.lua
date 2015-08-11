require 'nn'
require 'nngraph'
require 'optim'

unigrid = require 'unigrid'

cmd = torch.CmdLine()
cmd:text('Train a 1 dimensional grid lstm')
cmd:text('Options')
cmd:option('-size', 200,'rnn size')
cmd:option('-k', 15, 'size of input bit vector')
cmd:option('-n', 5, 'number of layers')
cmd:option('-mb', 64, 'minibatch size')
cmd:option('-iters', 100, 'number of iterations')
opt = cmd:parse(arg)

local x = nn.Identity()()

local h0 = nn.Linear(opt.k, opt.size)(x)
local m0 = nn.Linear(opt.k, opt.size)(x)
local hn, mn = unigrid.unigrid(opt.size, opt.n, h0, m0)


local y = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(opt.size, 1)(hn), 
    nn.Linear(opt.size, 1)(mn)
}))

local model = nn.gModule({x}, {y})
local criterion = nn.BCECriterion()

w, dw = model:getParameters()
w:uniform(-0.08, 0.08)
optim_state = {learningRate=1.}
feval = function(w_new)
    dw:zero()
    local xx = torch.round(torch.rand(opt.mb, opt.k))
    local yy = torch.sum(xx, 2):apply(function(i) return math.fmod(i, 2)end)
    local loss = criterion:forward(model:forward(xx), yy)
    model:backward(xx, criterion:backward(model.output, yy))
    return loss, dw
end

for i=1,opt.iters do
    _, fs = optim.sgd(feval,w, optim_state)
    print (torch.exp(fs[1]))
end 
