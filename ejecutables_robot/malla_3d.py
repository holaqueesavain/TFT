import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('output/mapeo_165725.csv', sep=';')
x, y, z = df['Robot_X'].values, df['Robot_Y'].values, df['Robot_Z'].values

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

color_puntos = '#219EBC'
color_lineas = '#8ECAE6'
color_texto  = '#023047'

ax.scatter(x, y, z, c=color_puntos, s=150, edgecolors='white', linewidth=1.5, alpha=1.0)

if len(x) == 27:
    Xg, Yg, Zg = x.reshape(3, 3, 3), y.reshape(3, 3, 3), z.reshape(3, 3, 3)
    for i in range(3):
        for j in range(3):
            ax.plot(Xg[:, i, j], Yg[:, i, j], Zg[:, i, j], color=color_lineas, alpha=0.8, linewidth=2)
            ax.plot(Xg[i, :, j], Yg[i, :, j], Zg[i, :, j], color=color_lineas, alpha=0.8, linewidth=2)
            ax.plot(Xg[i, j, :], Yg[i, j, :], Zg[i, j, :], color=color_lineas, alpha=0.8, linewidth=2)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(color='lightgray', linestyle=':', alpha=0.6)

ax.set_xlabel("Eje X (m)", color='gray', labelpad=10)
ax.set_ylabel("Eje Y (m)", color='gray', labelpad=10)
ax.set_zlabel("Eje Z (m)", color='gray', labelpad=10)

plt.savefig('malla_espacial.png', dpi=300, bbox_inches='tight')