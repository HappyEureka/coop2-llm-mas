import functools

import numpy as np
import opensimplex

from . import constants
from . import objects

def generate_world(world, mode):
  simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
  
  tunnels = np.zeros(world.area, bool)
  for x in range(world.area[0]):
    for y in range(world.area[1]):
      _set_material(world, (x, y), tunnels, simplex, mode)
  for x in range(world.area[0]):
    for y in range(world.area[1]):
      _set_object(world, (x, y), tunnels, mode)


def _set_material(world, pos, tunnels, simplex, mode):
  x, y = pos
  simplex = functools.partial(_simplex, simplex)
  uniform = world.random.uniform
  start = 4 - np.sqrt((x - world.area[0]//2) ** 2 + (y - world.area[1]//2) ** 2)
  start += 2 * simplex(x, y, 8, 3)
  start = 1 / (1 + np.exp(-start))
  water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
  water -= 2 * start
  mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
  mountain -= 4 * start + 0.3 * water
  if mode == 'grassland':
    if start > 0.5:
      world[x, y] = 'grass'
    else:
      if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
        world[x, y] = 'tree'
      else:
        world[x, y] = 'grass'
  else:
    if start > 0.5:
      world[x, y] = 'grass'
    elif mountain > 0.15:
      if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
        world[x, y] = 'path'
      elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
        world[x, y] = 'path'
        tunnels[x, y] = True
      elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
        world[x, y] = 'path'
        tunnels[x, y] = True
      elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
        world[x, y] = 'coal'
      elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
        world[x, y] = 'iron'
      elif mountain > 0.18 and uniform() > 0.994:
        world[x, y] = 'diamond'
      elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35:
        world[x, y] = 'lava'
      else:
        world[x, y] = 'stone'
    elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
      world[x, y] = 'sand'
    elif 0.3 < water:
      world[x, y] = 'water'
    else:  # grassland
      if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
        world[x, y] = 'tree'
      else:
        world[x, y] = 'grass'

def _set_object(world, pos, tunnels, mode):
  x, y = pos
  uniform = world.random.uniform
  dist = np.sqrt((x - world.area[0]//2) ** 2 + (y - world.area[1]//2) ** 2)
  material, _ = world[x, y]
  if material not in constants.walkable:
    pass
  elif dist > 3 and material == 'grass' and uniform() > 0.985:
    if mode != 'grassland':
      world.add(objects.Cow(world, (x, y)))



# import numpy as np
# import functools
# import opensimplex

# def generate_world(world, players):
#     simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
    
#     print('===============================================world.area:', world.area)
#     tunnels = np.zeros(world.area, bool)
    
#     for x in range(world.area[0]):
#         for y in range(world.area[1]):
#             _set_material(world, (x, y), players, tunnels, simplex)
    
#     for x in range(world.area[0]):
#         for y in range(world.area[1]):
#             _set_object(world, (x, y), players, tunnels)

# def _set_material(world, pos, players, tunnels, simplex):
#     x, y = pos
#     simplex = functools.partial(_simplex, simplex)
#     uniform = world.random.uniform
    
#     # Compute start based on all players' positions
#     start = max(4 - np.min([np.sqrt((x - p.pos[0]) ** 2 + (y - p.pos[1]) ** 2) for p in players]), 0)
#     start += 2 * simplex(x, y, 8, 3)
#     start = 1 / (1 + np.exp(-start))
    
#     water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
#     water -= 2 * start
#     mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
#     mountain -= 4 * start + 0.3 * water
    
#     if start > 0.5:
#         world[x, y] = 'grass'
#     elif mountain > 0.15:
#         if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
#             world[x, y] = 'path'
#         elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizontal tunnel
#             world[x, y] = 'path'
#             tunnels[x, y] = True
#         elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnel
#             world[x, y] = 'path'
#             tunnels[x, y] = True
#         elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
#             world[x, y] = 'coal'
#         elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
#             world[x, y] = 'iron'
#         elif mountain > 0.18 and uniform() > 0.994:
#             world[x, y] = 'diamond'
#         elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35:
#             world[x, y] = 'lava'
#         else:
#             world[x, y] = 'stone'
#     elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
#         world[x, y] = 'sand'
#     elif 0.3 < water:
#         world[x, y] = 'water'
#     else:  # grassland
#         if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
#             world[x, y] = 'tree'
#         else:
#             world[x, y] = 'grass'

# def _set_object(world, pos, players, tunnels):
#     x, y = pos
#     uniform = world.random.uniform
    
#     # Compute min distance from any player
#     dist = np.min([np.sqrt((x - p.pos[0]) ** 2 + (y - p.pos[1]) ** 2) for p in players])
#     material, _ = world[x, y]
    
#     if material not in constants.walkable:
#         return
    
#     # Spawn cows only if far enough from all players
#     if dist > 3 and material == 'grass' and uniform() > 0.985:
#         world.add(objects.Cow(world, (x, y)))

  # elif dist > 10 and uniform() > 0.993:
  #  world.add(objects.Zombie(world, (x, y), player))
  # elif material == 'path' and tunnels[x, y] and uniform() > 0.95:
  #  world.add(objects.Skeleton(world, (x, y), player))

def _simplex(simplex, x, y, z, sizes, normalize=True):
  if not isinstance(sizes, dict):
    sizes = {sizes: 1}
  value = 0
  for size, weight in sizes.items():
    if hasattr(simplex, 'noise3d'):
      noise = simplex.noise3d(x / size, y / size, z)
    else:
      noise = simplex.noise3(x / size, y / size, z)
    value += weight * noise
  if normalize:
    value /= sum(sizes.values())
  return value