import pygame
import numpy as np
import heapq
import random
import time
import sys

GRID_SIZE          = 30        
CELL_SIZE          = 22          
STATUS_BAR_H       = 60         
MARGIN             = 2          
 
WIN_W = GRID_SIZE * CELL_SIZE
WIN_H = GRID_SIZE * CELL_SIZE + STATUS_BAR_H
 
NUM_STATIC          = 120
NUM_DYNAMIC         = 90
DRONE_MOVE_DELAY    = 80       
DYNAMIC_MOVE_EVERY  = 8         
 
START  = (0, 0)
TARGET = (GRID_SIZE - 1, GRID_SIZE - 1)

C_BG         = (18,  18,  28)
C_GRID_LINE  = (30,  30,  45)
C_EMPTY      = (28,  28,  42)
C_OBSTACLE   = (180, 40,  50)   
C_DYNAMIC    = (220, 110, 40)    
C_PATH       = (40,  80,  160)   
C_VISITED    = (35,  55,  75)    
C_DRONE      = (0,   210, 255)   
C_TARGET     = (50,  200, 80)    
C_START      = (200, 200, 60)    
C_TEXT       = (200, 200, 210)
C_TEXT_HI    = (0,   210, 255)
C_STATUS_BG  = (12,  12,  22)
C_SUCCESS    = (50,  220, 100)
C_WARN       = (255, 160, 40)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def astar(blocked, start, goal):
    """
    blocked: set of (r,c) cells that are impassable.
    Returns path list [(r,c), ...] or None.
    """
    heap = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g = {start: 0}
    visited = set()
 
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]
        if cur in visited:
            continue
        visited.add(cur)
        r, c = cur
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
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
        self.static_obs  = self._place_static()
        self.dyn_obs     = self._place_dynamic()
        self.drone       = list(START)
        self.visited     = []
        self.path        = []
        self.path_idx    = 0
        self.steps       = 0
        self.reroutes    = 0
        self.paused      = False
        self.reached     = False
        self.no_path     = False
        self.dyn_counter = 0
        self.start_time  = time.time()
        self._compute_path()
    def _place_static(self):
        obs = set()
        attempts = 0
        while len(obs) < NUM_STATIC and attempts < 5000:
            r = random.randint(0, GRID_SIZE-1)
            c = random.randint(0, GRID_SIZE-1)
            if (r,c) not in (START, TARGET):
                obs.add((r,c))
            attempts += 1
        return obs
 
    def _place_dynamic(self):
        positions = []
        for _ in range(NUM_DYNAMIC):
            for _ in range(1000):
                r = random.randint(1, GRID_SIZE-2)
                c = random.randint(1, GRID_SIZE-2)
                if (r,c) not in self.static_obs and (r,c) not in (START, TARGET):
                    dr, dc = random.choice([(-1,0),(1,0),(0,-1),(0,1)])
                    positions.append({'pos':(r,c), 'dir':(dr,dc)})
                    break
        return positions

    def _blocked_set(self):
        blocked = set(self.static_obs)
        for d in self.dyn_obs:
            blocked.add(d['pos'])
        return blocked
 
    def _compute_path(self):
        blocked = self._blocked_set()
        path = astar(blocked, tuple(self.drone), TARGET)
        if path:
            self.path     = path
            self.path_idx = 0
            self.no_path  = False
        else:
            self.no_path = True
 
    def step(self):
        if self.reached or self.paused:
            return

        self.dyn_counter += 1
        if self.dyn_counter % DYNAMIC_MOVE_EVERY == 0:
            self._move_dynamic()
 
        if self.no_path:
            self._compute_path()
            return
 
        if self.path_idx >= len(self.path):
            self.reached = True
            return
 
        nxt = self.path[self.path_idx]

        blocked = self._blocked_set()
        if nxt in blocked:
            self.reroutes += 1
            self._compute_path()
            return

        self.visited.append(tuple(self.drone))
        self.drone = list(nxt)
        self.path_idx += 1
        self.steps += 1
 
        if tuple(self.drone) == TARGET:
            self.reached = True
 
    def _move_dynamic(self):
        occupied = set(self.static_obs) | {tuple(self.drone)} | {TARGET} | {START}
        for d in self.dyn_obs:
            r, c = d['pos']
            dr, dc = d['dir']
            nr, nc = r+dr, c+dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or (nr,nc) in occupied:
            
                dr, dc = -dr, -dc
                nr, nc = r+dr, c+dc
            if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) and (nr,nc) not in occupied:
                d['pos'] = (nr, nc)
                d['dir'] = (dr, dc)
 
    def efficiency(self):
        opt = heuristic(START, TARGET)
        if self.steps == 0:
            return 0.0
        return min(100.0, opt / self.steps * 100)
 
    def elapsed(self):
        return time.time() - self.start_time
 
    def path_len(self):
        return len(self.path) - self.path_idx if self.path else 0
 
def draw_cell(surf, r, c, color, shrink=0):
    x = c * CELL_SIZE + shrink
    y = r * CELL_SIZE + shrink
    s = CELL_SIZE - MARGIN - shrink * 2
    pygame.draw.rect(surf, color, (x, y, s, s), border_radius=3)
 
 
def draw(surf, sim, font_sm, font_md):
    surf.fill(C_BG)
 
    dyn_pos  = {d['pos'] for d in sim.dyn_obs}
    drone_rc = tuple(sim.drone)
    path_set = set(sim.path[sim.path_idx:]) if sim.path else set()
 
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rc = (r, c)
            if rc in sim.static_obs:
                draw_cell(surf, r, c, C_OBSTACLE)
            elif rc in dyn_pos:
                draw_cell(surf, r, c, C_DYNAMIC)
            elif rc == TARGET:
                draw_cell(surf, r, c, C_TARGET)
            elif rc == START:
                draw_cell(surf, r, c, C_START, shrink=3)
            elif rc in path_set:
                draw_cell(surf, r, c, C_PATH, shrink=4)
            elif rc in sim.visited:
                draw_cell(surf, r, c, C_VISITED, shrink=2)
            else:
                draw_cell(surf, r, c, C_EMPTY)

    dr, dc = sim.drone
    cx = dc * CELL_SIZE + CELL_SIZE // 2 - MARGIN // 2
    cy = dr * CELL_SIZE + CELL_SIZE // 2 - MARGIN // 2
    pygame.draw.circle(surf, (*C_DRONE, 60), (cx, cy), CELL_SIZE // 2 - 1)
    pygame.draw.circle(surf, C_DRONE, (cx, cy), CELL_SIZE // 2 - 3)
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 3)

    bar_y = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(surf, C_STATUS_BG, (0, bar_y, WIN_W, STATUS_BAR_H))
    pygame.draw.line(surf, C_DRONE, (0, bar_y), (WIN_W, bar_y), 1)
 
    # Status text
    if sim.reached:
        status_str = "Status: Goal reached!"
        status_col = C_SUCCESS
    elif sim.paused:
        status_str = "Status: PAUSED"
        status_col = C_WARN
    elif sim.no_path:
        status_str = "Status: Rerouting..."
        status_col = C_WARN
    else:
        status_str = "Status: Navigating"
        status_col = C_TEXT_HI
 
    line1 = (f"Time: {sim.elapsed():.1f}s    "
              f"Steps: {sim.steps}    "
              f"Reroutes: {sim.reroutes}    "
              f"Path Length: {sim.path_len()}    "
              f"Efficiency: {sim.efficiency():.1f}%")
 
    line2 = f"{status_str}    |    SPACE=pause   R=reset   ESC=quit"
 
    t1 = font_sm.render(line1, True, C_TEXT)
    t2 = font_sm.render(line2, True, status_col)
    surf.blit(t1, (10, bar_y + 8))
    surf.blit(t2, (10, bar_y + 32))

def main():
    pygame.init()
    pygame.display.set_caption("Autonomous Drone Navigation Simulation")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()
 
    try:
        font_sm = pygame.font.SysFont("Consolas", 13)
        font_md = pygame.font.SysFont("Consolas", 16, bold=True)
    except:
        font_sm = pygame.font.SysFont("monospace", 13)
        font_md = pygame.font.SysFont("monospace", 16, bold=True)
 
    sim = Sim()
    last_step = pygame.time.get_ticks()
 
    print("Drone simulation started.")
    print("SPACE = pause/resume | R = reset | ESC = quit")
 
    while True:
        clock.tick(60)
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    random.seed()   # new random layout
                    sim.reset()
 
        # Step drone on timer
        now = pygame.time.get_ticks()
        if now - last_step >= DRONE_MOVE_DELAY:
            sim.step()
            last_step = now
 
        draw(screen, sim, font_sm, font_md)
        pygame.display.flip()
 
    pygame.quit()
 
 
if __name__ == "__main__":
    random.seed(42)
    main()