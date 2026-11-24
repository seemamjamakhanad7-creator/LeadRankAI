from collections import deque
import heapq, itertools, random, math

# A small transparent search-planning example: choose feature bundles
# that reach a target information value at minimum cost.
FEATURES = {
    "Website activity": (3, 2),
    "Email engagement": (2, 1),
    "Demo request": (5, 3),
    "Company profile": (2, 2),
    "Budget fit": (4, 3),
}
TARGET = 8

def _neighbors(state):
    for name, (value, cost) in FEATURES.items():
        if name not in state:
            yield tuple(sorted((*state, name))), value, cost

def _value(state):
    return sum(FEATURES[x][0] for x in state)

def search_demo(method="BFS"):
    start = tuple()
    frontier = deque([(start, 0)]) if method in ("BFS","DFS") else [(0, 0, start)]
    seen = {start}
    expanded = 0
    counter = itertools.count()
    while frontier:
        if method == "BFS":
            state, cost = frontier.popleft()
        elif method == "DFS":
            state, cost = frontier.pop()
        else:
            cost, _, state = heapq.heappop(frontier)
        expanded += 1
        if _value(state) >= TARGET:
            return {"method":method,"features":list(state),"value":_value(state),
                    "cost":cost,"expanded":expanded}
        for nxt, value, step_cost in _neighbors(state):
            if nxt in seen: continue
            seen.add(nxt)
            new_cost = cost + step_cost
            if method == "BFS": frontier.append((nxt,new_cost))
            elif method == "DFS": frontier.append((nxt,new_cost))
            else: heapq.heappush(frontier,(new_cost,next(counter),nxt))
    return {"method":method,"features":[],"value":0,"cost":0,"expanded":expanded}

def greedy_demo():
    chosen=[]; value=0; cost=0
    remaining=dict(FEATURES)
    while value<TARGET and remaining:
        name,(v,c)=max(remaining.items(),key=lambda item:item[1][0]/item[1][1])
        chosen.append(name); value+=v; cost+=c; remaining.pop(name)
    return {"method":"Greedy","features":chosen,"value":value,"cost":cost}

def astar_demo():
    # A* with admissible h=0, equivalent to uniform-cost search for this toy problem.
    return search_demo("UCS") | {"method":"A* (h=0)"}

def assign_greedy(leads, reps):
    loads={r:0 for r in reps}
    assignments=[]
    for lead in sorted(leads,key=lambda x:x.get("score",0),reverse=True):
        rep=min(loads,key=loads.get)
        assignments.append((lead["id"],rep))
        loads[rep]+=1
    return assignments, loads
