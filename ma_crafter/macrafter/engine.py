import collections
import functools
import pathlib

import imageio.v3 as imageio
import numpy as np
from PIL import Image, ImageEnhance
from numba import jit, njit

class AttrDict(dict):
  __getattr__ = dict.__getitem__


class staticproperty:

  def __init__(self, function):
    self.function = function

  def __get__(self, instance, owner=None):
    return self.function()


class World:

  def __init__(self, area, materials, chunk_size):
    self.area = area
    self._chunk_size = chunk_size
    self._mat_names = {i: x for i, x in enumerate([None] + materials)}
    self._mat_ids = {x: i for i, x in enumerate([None] + materials)}
    self.reset()

  def reset(self, seed=None):
    self.random = np.random.RandomState(seed)
    self.daylight = 0.0
    self._chunks = collections.defaultdict(set)
    self._objects = [None]
    self._mat_map = np.zeros(self.area, np.uint8)
    self._obj_map = np.zeros(self.area, np.uint32)

  @property
  def objects(self):
    # Return a new list so the objects cannot change while being iterated over.
    return [obj for obj in self._objects if obj]

  @property
  def chunks(self):
    return self._chunks.copy()

  def add(self, obj):
    assert hasattr(obj, 'pos')
    obj.pos = np.array(obj.pos)
    assert self._obj_map[tuple(obj.pos)] == 0
    index = len(self._objects)
    self._objects.append(obj)
    self._obj_map[tuple(obj.pos)] = index
    self._chunks[self.chunk_key(obj.pos)].add(obj)

  def remove(self, obj):
    if obj.removed:
      return
    self._objects[self._obj_map[tuple(obj.pos)]] = None
    self._obj_map[tuple(obj.pos)] = 0
    self._chunks[self.chunk_key(obj.pos)].remove(obj)
    obj.removed = True

  def move(self, obj, pos):
    if obj.removed:
      return
    pos = np.array(pos)
    assert self._obj_map[tuple(pos)] == 0
    index = self._obj_map[tuple(obj.pos)]
    self._obj_map[tuple(pos)] = index
    self._obj_map[tuple(obj.pos)] = 0
    old_chunk = self.chunk_key(obj.pos)
    new_chunk = self.chunk_key(pos)
    if old_chunk != new_chunk:
      self._chunks[old_chunk].remove(obj)
      self._chunks[new_chunk].add(obj)
    obj.pos = pos

  def __setitem__(self, pos, material):
    if material not in self._mat_ids:
      id_ = len(self._mat_ids)
      self._mat_ids[material] = id_
    self._mat_map[tuple(pos)] = self._mat_ids[material]

  def __getitem__(self, pos):
    if not _inside((0, 0), pos, self.area):
      return None, None
    material = self._mat_names[self._mat_map[tuple(pos)]]
    obj = self._objects[self._obj_map[tuple(pos)]]
    return material, obj

  def nearby(self, pos, distance):
    (x, y), d = pos, distance
    ids = set(self._mat_map[
        x - d: x + d + 1, y - d: y + d + 1].flatten().tolist())
    materials = tuple(self._mat_names[x] for x in ids)
    indices = self._obj_map[
        x - d: x + d + 1, y - d: y + d + 1].flatten().tolist()
    objs = {self._objects[i] for i in indices if i > 0}
    return materials, objs

  def mask(self, xmin, xmax, ymin, ymax, material):
    region = self._mat_map[xmin: xmax, ymin: ymax]
    return (region == self._mat_ids[material])

  def count(self, material):
    return (self._mat_map == self._mat_ids[material]).sum()

  def chunk_key(self, pos):
    (x, y), (csx, csy) = pos, self._chunk_size
    xmin, ymin = (x // csx) * csx, (y // csy) * csy
    xmax = min(xmin + csx, self.area[0])
    ymax = min(ymin + csy, self.area[1])
    return (xmin, xmax, ymin, ymax)

@functools.lru_cache(100)
class Textures:
  def __init__(self, directory):
    self._originals = {}
    self._textures = {}
    for filename in pathlib.Path(directory).glob('*.png'):
      image = imageio.imread(filename.read_bytes())
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      self._originals[filename.stem] = image
      self._textures[(filename.stem, image.shape[:2])] = image

  def get(self, name, size):
    if name is None:
      name = 'unknown'
    # make sure size is at least 1x1
    size = max(1, int(size[0])), max(1, int(size[1]))

    key = name, size
    if key not in self._textures:
      image = self._originals[name]
      image = Image.fromarray(image)
      # print("Resizing", image.size, size)
      image = image.resize(size[::-1], resample=Image.NEAREST)
      image = np.array(image)
      self._textures[key] = image
    return self._textures[key]


class GlobalView:

  pass


class UncoverView:

  pass


class LocalView:

  def __init__(self, world, textures, grid):
    self._world = world
    self._textures = textures
    self._grid = np.array(grid)
    self._offset = (self._grid) // 2
    self._area = np.array(self._world.area)
    self._center = None
    self.visible_canvas = None

  # def __call__(self, player, unit):
  #   self._unit = np.array(unit)
  #   self._center = np.array(player.pos)
  #   canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8) + 127
  #   for x in range(self._grid[0]):
  #     for y in range(self._grid[1]):
  #       pos = self._center + np.array([x, y]) - self._offset
  #       if not _inside((0, 0), pos, self._area):
  #         continue
  #       texture = self._textures.get(self._world[pos][0], unit)
  #       _draw(canvas, np.array([x, y]) * unit, texture)
  #   for obj in self._world.objects:
  #     pos = obj.pos - self._center + self._offset
  #     if not _inside((0, 0), pos, self._grid):
  #       continue
  #     texture = self._textures.get(obj.texture, unit)
  #     _draw_alpha(canvas, pos * unit, texture)
  #   #canvas = self._light(canvas, self._world.daylight)
  #   if player.sleeping:
  #     canvas = self._sleep(canvas)
  #   if player.health < 1:
  #     canvas = self._tint(canvas, (128, 0, 0), 0.6)
  #   return canvas
  
  def __call__(self, players, unit):
    self._unit = np.array(unit)
    self.visible_canvas = np.zeros(tuple(self._area * unit) + (3,), np.uint8) + 127
    visible_mask = np.zeros(tuple(self._area), dtype=bool)

    # Mark visible areas for all players
    for player in players:
      center = np.array(player.pos)
      top_left = np.maximum(center - self._offset, [0, 0])
      bottom_right = np.minimum(center + self._offset, self._area - 1)

      xs = np.arange(top_left[0], bottom_right[0] + 1)
      ys = np.arange(top_left[1], bottom_right[1] + 1)
      visible_mask[np.ix_(xs, ys)] = True

    # Draw tiles once over visible area
    for x in range(self._area[0]):
      for y in range(self._area[1]):
        if not visible_mask[x, y]:
          continue
        texture = self._textures.get(self._world[x, y][0], unit)
        _draw(self.visible_canvas, np.array([x, y]) * unit, texture)

    # Draw objects once if inside any visible area
    for obj in self._world.objects:
      pos = obj.pos
      if not _inside((0, 0), pos, self._area):
        continue
      if not visible_mask[pos[0], pos[1]]:
        continue
      texture = self._textures.get(obj.texture, unit)
      _draw_alpha(self.visible_canvas, pos * unit, texture)

    # Apply per-player effects globally (optional logic — if you want per-player overlays, this should be done elsewhere)
    for player in players:
      if player.sleeping:
        self.visible_canvas = self._sleep(self.visible_canvas)
      if player.health < 1:
        self.visible_canvas = self._tint(self.visible_canvas, (128, 0, 0), 0.6)
    
    # print("Visible canvas shape:", self.visible_canvas.shape)
    return self
  
  def slice_player_view(self, player):
    center = np.array(player.pos)
    unit = self._unit
    expected_size = tuple(self._grid * unit) + (3,)

    top_left = (center - self._offset) * unit
    bottom_right = (center + self._offset + 1) * unit

    # Clamp coordinates to canvas boundary
    canvas_h, canvas_w = self.visible_canvas.shape[:2]
    clamped_top_left = np.maximum(top_left, [0, 0])
    clamped_bottom_right = np.minimum(bottom_right, [canvas_w, canvas_h])

    # Slice the visible canvas (may be smaller)
    sliced = self.visible_canvas[
      clamped_top_left[0]:clamped_bottom_right[0],
      clamped_top_left[1]:clamped_bottom_right[1]
    ]

    # Prepare the output with default background (127 gray)
    result = np.zeros(expected_size, dtype=np.uint8) + 127

    # Calculate where to paste the sliced image inside the result
    paste_x = clamped_top_left[0] - top_left[0]
    paste_y = clamped_top_left[1] - top_left[1]

    h, w = sliced.shape[:2]
    result[paste_x:paste_x + h, paste_y:paste_y + w] = sliced
    return result

  # def _light(self, canvas, daylight):
  #   night = canvas
  #   if daylight < 0.5:
  #     night = self._noise(night, 2 * (0.5 - daylight), 0.5)
  #   night = np.array(ImageEnhance.Color(
  #       Image.fromarray(night.astype(np.uint8))).enhance(0.4))
  #   night = self._tint(night, (0, 16, 64), 0.5)
  #   return daylight * canvas + (1 - daylight) * night

  def _sleep(self, canvas):
    canvas = np.array(ImageEnhance.Color(
       Image.fromarray(canvas.astype(np.uint8))).enhance(0.0))
    canvas = self._tint(canvas, (0, 0, 16), 0.5)
    return canvas

  def _tint(self, canvas, color, amount):
    color = np.array(color)
    return (1 - amount) * canvas + amount * color

  def _noise(self, canvas, amount, stddev):
    noise = self._world.random.uniform(32, 127, canvas.shape[:2])[..., None]
    mask = amount * self._vignette(canvas.shape, stddev)[..., None]
    return (1 - mask) * canvas + mask * noise
  
  @functools.lru_cache(10)
  @jit
  def _vignette(self, shape, stddev):
    xs, ys = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]))
    return 1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / (stddev ** 2)).T


class ItemView:

  def __init__(self, textures, grid):
    self._textures = textures
    self._grid = np.array(grid)

  def __call__(self, inventory, unit):
    unit = np.array(unit)
    canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8)
    for index, (item, amount) in enumerate(inventory.items()):
      if amount < 1:
        continue
      self._item(canvas, index, item, unit)
      self._amount(canvas, index, amount, unit)
    return canvas
  
  def _item(self, canvas, index, item, unit):
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * unit + 0.1 * unit).astype(np.int32)
    texture = self._textures.get(item, 0.8 * unit)
    _draw_alpha(canvas, pos, texture)
  
  def _amount(self, canvas, index, amount, unit):
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * unit + 0.4 * unit).astype(np.int32)
    text = str(amount) if amount in list(range(10)) else 'unknown'
    texture = self._textures.get(text, 0.6 * unit)
    _draw_alpha(canvas, pos, texture)


class SemanticView:

  def __init__(self, world, obj_types):
    self._world = world
    self._mat_ids = world._mat_ids.copy()
    self._obj_ids = {
        c: len(self._mat_ids) + i
        for i, c in enumerate(obj_types)}
    self.obj_ids = {}
    self.obj_id = 0
  def __call__(self, return_ids=False):
    canvas = self._world._mat_map.copy()
    #ids = np.zeros(canvas.shape, np.uint64)
    ids = np.arange(canvas.size, dtype=np.uint64).reshape(canvas.shape)
    if return_ids:
      for obj in self._world.objects:
        canvas[tuple(obj.pos)] = self._obj_ids[type(obj)]
        #print(obj, obj.pos, type(obj), id(obj))
        if id(obj) not in self.obj_ids:
          self.obj_ids[id(obj)] = self.obj_id
          self.obj_id += 1
        ids[tuple(obj.pos)] = self.obj_ids[id(obj)]
      return canvas, ids
    else:
      for obj in self._world.objects:
        canvas[tuple(obj.pos)] = self._obj_ids[type(obj)]
      return canvas

@njit
def _inside(lhs, mid, rhs):
  return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])

@njit
def _draw(canvas, pos, texture):
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    texture = texture[..., :3]
  canvas[x: x + w, y: y + h] = texture
  
@njit
def _draw_alpha(canvas, pos, texture):
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    alpha = texture[..., 3:] / 255 #.astype(np.float32) / 255
    texture = texture[..., :3] / 255 #.astype(np.float32) / 255
    current = canvas[x: x + w, y: y + h] / 255#.astype(np.float32) / 255
    blended = alpha * texture + (1 - alpha) * current
    texture = (255 * blended).astype(np.uint8)
  canvas[x: x + w, y: y + h] = texture
