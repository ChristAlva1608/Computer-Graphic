from libs.shader import *
from libs.buffer import *
import glm
import numpy as np
import math

class Floor:
    def __init__(self, vert_shader, frag_shader, width=10.0, depth=10.0):
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()
        
        self.width = width
        self.depth = depth
        
        # Create a large rectangle for the floor
        self.vertices = np.array([
            # Position         # Normal         
            -width, 0.0, -depth,  0.0, 1.0, 0.0,  # Bottom-left
             width, 0.0, -depth,  0.0, 1.0, 0.0,  # Bottom-right
             width, 0.0,  depth,  0.0, 1.0, 0.0,  # Top-right
            -width, 0.0,  depth,  0.0, 1.0, 0.0,  # Top-left
        ], dtype=np.float32)
        
        self.indices = np.array([
            0, 1, 2,  # First triangle
            0, 2, 3   # Second triangle
        ], dtype=np.uint32)
        
        # Gray color for the floor
        self.colors = np.array([
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5
        ], dtype=np.float32)

    def setup(self):
        # Set up vertex buffer objects
        self.vao.add_vbo(0, self.vertices.reshape(-1, 6)[:, 0:3], ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors.reshape(-1, 3), ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.vertices.reshape(-1, 6)[:, 3:6], ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Initial transformations
        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(5, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 0, 1))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model
        
        # Upload uniforms
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Light setup
        I_light = np.array([
            [0.9, 0.9, 0.9],  # diffuse
            [0.9, 0.9, 0.9],  # specular
            [0.3, 0.3, 0.3],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials setup
        K_materials = np.array([
            [0.5, 0.5, 0.5],  # diffuse
            [0.5, 0.5, 0.5],  # specular
            [0.5, 0.5, 0.5],  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(32.0, 'shininess')
        self.uma.upload_uniform_scalar1i(1, 'mode')

        return self

    def draw(self):
        GL.glUseProgram(self.shader.render_idx)
        
        model_view_matrix = self.view * self.model
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)