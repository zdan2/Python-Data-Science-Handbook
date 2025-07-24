from collections import defaultdict
import sys
sys.setrecursionlimit(10**6)
v=set()
far_p=-1
far_c=0
def dfs(g,p,c=0):
    global far_p,far_c
    if p not in v:
        v.add(p)
        for nxt in g[p]:
            dfs(g,nxt,c+1)
        if c>far_c:
            far_c=c
            far_p=p
n=int(input())
d=defaultdict(list)
for _ in range(n-1):
    a,b=map(int,input().split())
    d[a].append(b)
    d[b].append(a)
dfs(d,1)
v=set()
far_c=0
dfs(d,far_p)
print(far_c+1)