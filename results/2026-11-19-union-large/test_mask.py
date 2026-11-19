import torch, ulhook
WIN=4
def literal_port(ids, w, K, union):
    N, k8 = ids.shape; keep = [[False] * k8 for _ in range(N)]
    for s0 in range(0, N - N % WIN, WIN):
        rows = range(s0, s0 + WIN); present = set(); core = {}
        for t in rows:
            wv = w[t].tolist()
            for i in range(k8):
                r = sum((wv[j] > wv[i]) or (wv[j] == wv[i] and j < i) for j in range(k8))
                core[t, i] = r < K
                if core[t, i]: present.add(int(ids[t, i]))
        for t in rows:
            for i in range(k8):
                keep[t][i] = core[t, i] or (union and int(ids[t, i]) in present)
    for t in range(N - N % WIN, N):
        wv = w[t].tolist()
        for i in range(k8):
            keep[t][i] = sum((wv[j] > wv[i]) or (wv[j] == wv[i] and j < i) for j in range(k8)) < K
    return torch.tensor(keep)
class R: global_num_experts=16
g = torch.Generator().manual_seed(1); n=0
for trial in range(40):
    N = 4*13 + (trial % 4); k = [4,8][trial%2]
    lg = torch.randn(N,16,generator=g)*2
    if trial % 3 == 0: lg = lg.round()  # ties
    pr = lg.softmax(-1); wv, iv = torch.topk(pr, k, dim=-1)
    perm = torch.randperm(k, generator=g); wv, iv = wv[:, perm], iv[:, perm]  # unsorted slots
    pad = 5; wv = torch.cat([wv, torch.rand(pad,k)]); iv = torch.cat([iv, torch.randint(0,16,(pad,k))])
    for K in range(1, k+1):
        for un in (False, True):
            ulhook.ST.update(T=N, K=K, union=un, stats=torch.zeros(8, dtype=torch.float64))
            w2 = ulhook.apply_mask(R(), wv.bfloat16() if trial%5==0 else wv, iv.int())
            w_ = wv.bfloat16() if trial%5==0 else wv
            ref = literal_port(iv[:N], w_[:N].float(), K, un)
            assert torch.equal(w2[:N] != 0, ref) , (trial,K,un)
            assert torch.equal(w2[N:], w_[N:]); n+=1
print("PASS", n)
