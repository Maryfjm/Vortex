import numpy as np
from PIL import Image

from fluid import Fluid

FRAME_PATH = 'placeholder'

RESOLUTION = 700, 700
VISCOSITY = 10 ** -3

DURATION = 100

INFLOW_PADDING = 60
INFLOW_DURATION = 60
INFLOW_RADIUS = 8
INFLOW_VELOCITY = 3

DOTS= 12
LAYERS = 3
r = [0]*LAYERS

def circle(theta):
    return np.asarray((np.cos(theta), np.sin(theta)))

center = np.floor_divide(RESOLUTION, 2)

count = 0

r[count] = np.min(center) - INFLOW_PADDING*(count+1)
directions = tuple(-circle(p * np.pi * 2 / DOTS + np.pi/2) for p in range(DOTS))
points = tuple(r[count] * circle(p+1 * np.pi * 2 / DOTS) + center for p in range(DOTS))
count +=1
while count < LAYERS:

    r[count] = np.min(center) - INFLOW_PADDING*(count+1)
    directions += tuple(-circle(p * np.pi * 2 / DOTS + np.pi/2) for p in range(DOTS))
    points += tuple(r[count] * circle(p * np.pi * 2 / DOTS) + center for p in range(DOTS))
    count +=1
    
channels = 'r', 'g', 'b'
fluid = Fluid(RESOLUTION, VISCOSITY, channels)

inflow_dye_field = np.zeros((fluid.size, len(channels)))
inflow_velocity_field = np.zeros_like(fluid.velocity_field)
for i, p in enumerate(points):
    distance = np.linalg.norm(fluid.indices - p, axis=1)
    mask = distance <= INFLOW_RADIUS

    for d in range(2):
        inflow_velocity_field[..., d][mask] = directions[i][d] * INFLOW_VELOCITY

    inflow_dye_field[..., 1][mask] = 1

for frame in range(DURATION):
    print(f'Computing frame {frame}.')

    fluid.advect_diffuse()

    if frame <= INFLOW_DURATION:
        fluid.velocity_field += inflow_velocity_field

        for i, k in enumerate(channels):
            fluid.quantities[k] += inflow_dye_field[..., i]

    fluid.project()

    rgb = np.dstack(tuple(fluid.quantities[c] for c in channels))

    rgb = rgb.reshape((*RESOLUTION, 3))
    rgb = (np.clip(rgb, 0, 1) * 255).astype('uint8')
    Image.fromarray(rgb).save(f'{FRAME_PATH}Frame {frame}.png')

