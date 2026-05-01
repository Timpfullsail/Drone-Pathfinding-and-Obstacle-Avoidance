import pygame
import heapq
import random
import time
import sys
import math
 

GRID_SIZE          = 20
CELL_SIZE          = 20
PANEL_W            = 160
MARGIN             = 2

WIN_W              = GRID_SIZE * CELL_SIZE + PANEL_W
WIN_H              = GRID_SIZE * CELL_SIZE

NUM_STATIC         = 60
NUM_DYNAMIC        = 20
DRONE_MOVE_DELAY   = 70
DYNAMIC_MOVE_EVERY = 2
DIR_CHANGE_CHANCE  = 0.20
START              = (0, 0)
TARGET             = (GRID_SIZE - 1, GRID_SIZE - 1)

ALL_DIRS  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
CARD_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

C_BG         = (8,   14,  8)
C_GRID       = (18,  35,  18)
C_EMPTY      = (12,  22,  12)
C_OBSTACLE   = (20,  160, 60)
C_DYNAMIC    = (180, 220, 40)
C_PATH       = (0,   255, 120)
C_VISITED    = (10,  55,  25)
C_DRONE      = (255, 255, 255)
C_DRONE_FILL = (30,  200, 100)
C_TARGET     = (255, 60,  60)
C_START      = (60,  120, 255)
C_PANEL_BG   = (5,   12,  5)
C_PANEL_LINE = (0,   180, 70)
C_TEXT_DIM   = (60,  120, 60)
C_TEXT_MED   = (100, 200, 100)
C_TEXT_HI    = (0,   255, 100)
C_TEXT_WARN  = (220, 200, 20)
C_TEXT_OK    = (60,  255, 120)
C_TEXT_ERR   = (255, 80,  80)

def heuristic(a,b):
    return abs(a[0] - b[0]) + abs(a[1]- b[1])
 
def astar(blocked, start, goal):
    heap = [(heuristic(start, goal), 0, start)]
    came_from, g, visited = {}, {start: 0}, set()
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            return [start] + path[::-1]
        if cur in visited:
            continue
        visited.add(cur)
        r, c = cur
        for dr, dc in CARD_DIRS:
            nb = (r+dr, c+dc)
            if not (0 <= nb[0] < GRID_SIZE and 0 <= nb[1] < GRID_SIZE):
                continue
            if nb in blocked or nb in visited:
                continue
            ng = g[cur] + 1
            if ng < g.get(nb, 10**9):
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + heuristic(nb, goal), ng, nb))
    return None
class Sim:
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.static_obs = self._static()
        self.dyn_obs    = self._dynamic()
        self.drone      = list(START)
        self.visited    = []
        self.path       = []
        self.path_idx   = 0
        self.steps      = 0
        self.reroutes   = 0
        self.paused     = False
        self.reached    = False
        self.no_path    = False
        self.dyn_count  = 0
        self.start_time = time.time()
        self._replan()
 
    def _static(self):
        obs, attempts = set(), 0
        while len(obs) < NUM_STATIC and attempts < 10000:
            r, c = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            if (r, c) not in (START, TARGET):
                obs.add((r, c))
                if random.random() < 0.30:
                    dr, dc = random.choice(CARD_DIRS)
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in (START, TARGET):
                        obs.add((nr, nc))
            attempts += 1
        for cell in [(0,1),(1,0),(1,1),(GRID_SIZE-2,GRID_SIZE-1),(GRID_SIZE-1,GRID_SIZE-2)]:
            obs.discard(cell)
        return obs
 
    def _dynamic(self):
        out = []
        for _ in range(NUM_DYNAMIC):
            for _ in range(2000):
                r, c = random.randint(2, GRID_SIZE-3), random.randint(2, GRID_SIZE-3)
                if (r, c) not in self.static_obs and (r, c) not in (START, TARGET):
                    out.append({'pos': (r, c), 'dir': random.choice(ALL_DIRS), 'ttl': random.randint(3, 10)})
                    break
        return out
 
    def _blocked(self):
        b = set(self.static_obs)
        for d in self.dyn_obs:
            b.add(d['pos'])
        return b
 
    def _replan(self):
        p = astar(self._blocked(), tuple(self.drone), TARGET)
        if p:
            self.path, self.path_idx, self.no_path = p, 0, False
        else:
            self.no_path = True
 
    def step(self):
        if self.reached or self.paused:
            return
        self.dyn_count += 1
        if self.dyn_count % DYNAMIC_MOVE_EVERY == 0:
            self._move_dyn()
        if self.no_path:
            self._replan()
            return
        if self.path_idx >= len(self.path):
            self.reached = True
            return
        nxt = self.path[self.path_idx]
        if nxt in self._blocked():
            self.reroutes += 1
            self._replan()
            return
        self.visited.append(tuple(self.drone))
        self.drone = list(nxt)
        self.path_idx += 1
        self.steps += 1
        if tuple(self.drone) == TARGET:
            self.reached = True
 
    def _move_dyn(self):
        occ = self._blocked() | {tuple(self.drone), TARGET, START}
        for d in self.dyn_obs:
            d['ttl'] -= 1
            if d['ttl'] <= 0 or random.random() < DIR_CHANGE_CHANCE:
                d['dir'] = random.choice(ALL_DIRS)
                d['ttl'] = random.randint(3, 10)
            r, c = d['pos']
            dr, dc = d['dir']
            cands = [(dr, dc)] + [x for x in ALL_DIRS if x != (dr, dc)]
            random.shuffle(cands[1:])
            for tdr, tdc in cands:
                nr, nc = r+tdr, c+tdc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in occ:
                    d['pos'] = (nr, nc)
                    d['dir'] = (tdr, tdc)
                    break
 
    def efficiency(self):
        return min(100.0, heuristic(START, TARGET) / self.steps * 100) if self.steps else 0.0
 
    def elapsed(self):
        return time.time() - self.start_time
 
    def path_len(self):
        return len(self.path) - self.path_idx if self.path else 0
 
def draw_cell(surf, r, c, color, shrink=0):
    x = c * CELL_SIZE + shrink
    y = r * CELL_SIZE + shrink
    s = CELL_SIZE - MARGIN - shrink * 2
    pygame.draw.rect(surf, color, (x, y, s, s), border_radius=3)
 
def draw_diamond(surf, r, c, color, shrink=0):
    cx = c * CELL_SIZE + CELL_SIZE // 2
    cy = r * CELL_SIZE + CELL_SIZE // 2
    h = CELL_SIZE // 2 - MARGIN - shrink
    pts = [(cx, cy-h), (cx+h, cy), (cx, cy+h), (cx-h, cy)]
    pygame.draw.polygon(surf, color, pts)
 
def draw_circle(surf, r, c, color, shrink=0):
    cx = c * CELL_SIZE + CELL_SIZE // 2
    cy = r * CELL_SIZE + CELL_SIZE // 2
    radius = CELL_SIZE // 2 - MARGIN - shrink
    pygame.draw.circle(surf, color, (cx, cy), radius)
 
def draw_panel(surf, sim, fb, fs, tick):
    px = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(surf, C_PANEL_BG, (px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surf, C_PANEL_LINE, (px, 0), (px, WIN_H), 2)
    for y in range(0, WIN_H, 4):
        pygame.draw.line(surf, (0, 8, 0), (px, y), (px + PANEL_W, y))
 
    y = 16
    t = fb.render("[ DRONE NAV ]", True, C_TEXT_HI)
    surf.blit(t, (px + PANEL_W // 2 - t.get_width() // 2, y))
    y += 30
    pygame.draw.line(surf, C_PANEL_LINE, (px+8, y), (px+PANEL_W-8, y))
    y += 12
 
    dot_col = C_TEXT_OK if (tick // 500) % 2 == 0 else C_TEXT_DIM
    pygame.draw.circle(surf, dot_col, (px+14, y+7), 5)
    if sim.reached:
        stxt, scol = "GOAL REACHED", C_TEXT_OK
    elif sim.paused:
        stxt, scol = "PAUSED",       C_TEXT_WARN
    elif sim.no_path:
        stxt, scol = "REROUTING",    C_TEXT_WARN
    else:
        stxt, scol = "NAVIGATING",   C_TEXT_HI
    surf.blit(fb.render(stxt, True, scol), (px+24, y))
    y += 28
    pygame.draw.line(surf, C_PANEL_LINE, (px+8, y), (px+PANEL_W-8, y))
    y += 12
 
    def row(label, val, col=C_TEXT_MED):
        nonlocal y
        surf.blit(fs.render(label, True, C_TEXT_DIM), (px+10, y))
        v = fb.render(str(val), True, col)
        surf.blit(v, (px + PANEL_W - 10 - v.get_width(), y))
        y += 26
 
    row("TIME",       f"{sim.elapsed():.1f}s")
    row("STEPS",      sim.steps)
    row("REROUTES",   sim.reroutes, C_TEXT_ERR if sim.reroutes > 5 else C_TEXT_MED)
    row("PATH LEN",   sim.path_len())
    eff = sim.efficiency()
    row("EFFICIENCY", f"{eff:.1f}%", C_TEXT_OK if eff >= 70 else C_TEXT_WARN)
 
    y += 8
    pygame.draw.line(surf, C_PANEL_LINE, (px+8, y), (px+PANEL_W-8, y))
    y += 12
    for key, act in [("SPC", "pause"), ("R", "reset"), ("ESC", "quit")]:
        ln = fs.render(f"[{key}] {act}", True, C_TEXT_DIM)
        surf.blit(ln, (px + PANEL_W // 2 - ln.get_width() // 2, y))
        y += 15

def draw(surf, sim, fonts, tick):
    fb, fs = fonts
    surf.fill(C_BG)
 
    gw = GRID_SIZE * CELL_SIZE
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(surf, C_GRID, (i * CELL_SIZE, 0), (i * CELL_SIZE, WIN_H))
        pygame.draw.line(surf, C_GRID, (0, i * CELL_SIZE), (gw, i * CELL_SIZE))
 
    dyn_pos  = {d['pos'] for d in sim.dyn_obs}
    path_set = set(sim.path[sim.path_idx:]) if sim.path else set()
    vis_set  = set(sim.visited)
 
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rc = (r, c)
            if rc in sim.static_obs:
                draw_cell(surf, r, c, C_OBSTACLE)
            elif rc in dyn_pos:
                draw_diamond(surf, r, c, C_DYNAMIC)
            elif rc == TARGET:
                draw_cell(surf, r, c, C_TARGET)
            elif rc == START:
                draw_cell(surf, r, c, C_START, shrink=3)
            elif rc in path_set:
                draw_cell(surf, r, c, C_PATH, shrink=4)
            elif rc in vis_set:
                draw_cell(surf, r, c, C_VISITED, shrink=2)
            else:
                draw_cell(surf, r, c, C_EMPTY)

    dr, dc = sim.drone
    draw_circle(surf, dr, dc, C_DRONE_FILL)
    draw_circle(surf, dr, dc, C_DRONE, shrink=4)
    cx = dc * CELL_SIZE + CELL_SIZE // 2
    cy = dr * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 3)
 
    draw_panel(surf, sim, fb, fs, tick)

def main():
    pygame.init()
    pygame.display.set_caption("Autonomous Drone Navigation Simulation")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()
    try:
        fb = pygame.font.SysFont("Consolas", 13, bold=True)
        fs = pygame.font.SysFont("Consolas", 11)
    except:
        fb = pygame.font.SysFont("monospace", 13, bold=True)
        fs = pygame.font.SysFont("monospace", 11)
 
    sim = Sim()
    last_step = pygame.time.get_ticks()
    print("SPACE=pause | R=reset | ESC=quit")
 
    while True:
        tick = pygame.time.get_ticks()
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    random.seed()
                    sim.reset()
        if tick - last_step >= DRONE_MOVE_DELAY:
            sim.step()
            last_step = tick
        draw(screen, sim, (fb, fs), tick)
        pygame.display.flip()
    pygame.quit()
 
if __name__ == "__main__":
    random.seed()
    main()