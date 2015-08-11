local lstm = require 'lstm'
local unigrid = {}
function unigrid.unigrid(n, size)
    local prev_h = nn.Identity()()
    local prev_m = nn.Identity()()
    local state = {}
    state[1] = {prev_h, prev_m}
    for i=1,n do
        state[i+1] = lstm.lstm(size)(state[i])
    end
    return nn.gModule(state[1], state[n+1])
end
return unigrid
