import numpy as np
from isaacgym import gymtorch, gymapi, gymutil

class SingleLine(gymutil.LineGeometry):
    "many thanks to Dr. Fan Shi the great coder"
    def __init__(self, x=0.0, y=0.0, z=0.5, pose=None, c=None):
        verts = np.empty((3, 2), gymapi.Vec3.dtype)
        verts[0][0] = (0, 0, 0)
        verts[0][1] = (x, y, z)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(1, gymapi.Vec3.dtype)
        colors[0] = (1.0, 0.0, 0.0)
        if c: colors[0] = c
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors