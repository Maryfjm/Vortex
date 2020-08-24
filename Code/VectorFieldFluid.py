import numpy as np
from PIL import Image

from fluid import Fluid

import matplotlib.pyplot as plt


FRAME_PATH = 'placeholder'

RESOLUTION = 300,300 
VISCOSITY = 10 ** -5
DURATION = 150

INFLOW_PADDING = 50
INFLOW_DURATION = 1
INFLOW_RADIUS = 1
INFLOW_VELOCITY = 2


#def circle(theta):
    #return np.asarray((np.cos(theta), np.sin(theta)))

center = np.floor_divide(RESOLUTION, 2)
#r = np.min(center) - INFLOW_PADDING

x,y = np.meshgrid(np.linspace(-RESOLUTION[0]/2,RESOLUTION[0]/2,RESOLUTION[0]),np.linspace(-RESOLUTION[1]/2,RESOLUTION[1]/2,RESOLUTION[1]))
u = (-y/np.sqrt((x+20)**2 + y**2+1))*np.e**(-((x+20)**2 + y**2)/1000)-(y/np.sqrt((x-20)**2 + y**2+1))*np.e**(-((x-20)**2 + y**2)/1000)
v = ((x+20)/np.sqrt((x+20)**2 + y**2+1))*np.e**(-((x+20)**2 + y**2)/1000)+((x-20)/np.sqrt((x-20)**2 + y**2+1))*np.e**(-((x-20)**2 + y**2)/1000)


directions = tuple([v[t][p],u[t][p]] for t in range(RESOLUTION[1]) for p in range(RESOLUTION[0]))#tuple(-circle(p * np.pi * 1 / 6) for p in range(2))
points = tuple([y[t][p], x[t][p]]+center for t in range(RESOLUTION[1]) for p in range(RESOLUTION[0])) #tuple(r * circle(p * np.pi * 1 / 12) + center for p in range(2))

u_v =u**2+v**2
u_v = u_v/u_v.max()

plt.contourf(x, y, u_v, 20, cmap='RdGy')
plt.colorbar();
plt.show()
plt.figure(figsize=(200,150))
plt.quiver(x,y,u,v)
plt.show()

Energy = []

channels = 'r', 'g', 'b'
fluid = Fluid(RESOLUTION, VISCOSITY, channels)

inflow_dye_field = np.zeros((fluid.size, len(channels)))
inflow_velocity_field = np.zeros_like(fluid.velocity_field)
for i in range(RESOLUTION[0]*RESOLUTION[1]):
    #distance = np.linalg.norm(fluid.indices - p, axis=1)
    #mask = distance <= INFLOW_RADIUS

    for d in range(1):
        inflow_velocity_field[i][d]=directions[i][d] * INFLOW_VELOCITY#[mask]=directions[i][d] * INFLOW_VELOCITY

    #inflow_dye_field[..., 1][mask] = 1
inflow_dye_field[..., 1]= u_v.reshape(RESOLUTION[0]*RESOLUTION[1],)#[mask] = u_v.reshape(14400,)

for frame in range(DURATION):
    print(f'Computing frame {frame}.')

    fluid.advect_diffuse()
    Energy.append(np.sum(np.square(fluid.velocity_field)))

    if frame <= INFLOW_DURATION:
        fluid.velocity_field += inflow_velocity_field

        for i, k in enumerate(channels):
            fluid.quantities[k] += inflow_dye_field[..., i]

    fluid.project()

    rgb = np.dstack(tuple(fluid.quantities[c] for c in channels))

    rgb = rgb.reshape((*RESOLUTION, 3))
    rgb = (np.clip(rgb, 0, 1) * 255).astype('uint8')
    Image.fromarray(rgb).save(f'{FRAME_PATH}Frame {frame}.png')
    

plt.ylabel(r'$\frac{u^2}{u^2_{0}}$')
plt.xlabel("time (dimensionless)")    
plt.plot(Energy/max(Energy))

