import sys, math

def log(x):
    if x == 0: return float("-inf")
    else: return math.log(x)

# convert (frequency, val) pair to dictionary
def dictify(dist):
    new_dist = dict()
    for (f,v) in dist:
        new_dist[v] = f
    return new_dist

# convert dictionary distribution into not dictionary
def undictify(dist):
    new_dist = []
    for (v,f) in dist.keys():
        new_dist.append((f,v))
    return new_dist

def cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f,U1,U2):
    assert(len(obs_freqs) == len(f))
    aux1_dict = dictify(aux1)
    aux2_dict = dictify(aux2)
    s1 = 0
    for v in U1:
        if v in aux1_dict:
            s1 += aux1_dict[v]
    s2 = 0
    for v in U2:
        if v in aux2_dict:
            s2 += aux2_dict[v]
    
    # Check if s1 and s2 are valid values to prevent log(0) or log(negative numbers)
    s1 = max(s1, sys.float_info.min)
    s2 = max(s2, sys.float_info.min)
    
    p = u1 * log(s1) + u2 * log(s2)
    
    # Check if p is a valid value
    if math.isnan(p) or math.isinf(p):
        return float('-inf')
    
    for i in range(len(obs_freqs)):
        (c1,c2) = obs_freqs[i]
        if f[i] in aux1_dict and f[i] in aux2_dict:
            val1 = aux1_dict[f[i]]
            val2 = aux2_dict[f[i]]
            # Ensure values are positive to prevent log(0)
            val1 = max(val1, sys.float_info.min)
            val2 = max(val2, sys.float_info.min)
            term = c1*log(val1) + c2 * log(val2)
            # Check if the term is valid
            if not (math.isnan(term) or math.isinf(term)):
                p += term
    return p