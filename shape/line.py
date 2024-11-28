from libs.shader import *
from libs.buffer import *
import numpy as np
import glm

class Line:
    def __init__(self, vert_shader, frag_shader):
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        self.vertices = np.empty((0, 3), dtype=np.float32)  # Empty (x, y, z)
        self.colors = np.empty((0, 3), dtype=np.float32)
    
    def add_vertex(self, position, color=(1.0, 1.0, 1.0)):
        """
        Add a new vertex and color to the line.
        :param position: Tuple or list (x, y, z) for vertex position.
        :param color: Tuple or list (r, g, b) for vertex color. Defaults to white.
        """
        self.vertices = np.vstack([self.vertices, np.array(position, dtype=np.float32)])
        self.colors = np.vstack([self.colors, np.array(color, dtype=np.float32)])
        self.update_buffers()
    
    def reset_vertex(self):
        """
        Add a new vertex and color to the line.
        :param position: Tuple or list (x, y, z) for vertex position.
        :param color: Tuple or list (r, g, b) for vertex color. Defaults to white.
        """
        self.vertices = np.empty((0, 3), dtype=np.float32)  # Empty (x, y, z)
        self.colors = np.empty((0, 3), dtype=np.float32)
        self.update_buffers()

    def update_buffers(self):
        """
        Update the vertex buffer objects (VBOs) with the current vertices and colors.
        """
        # Bind and update the VBOs
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

    def setup(self):
        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model 

        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)
        return self

    def draw(self):
        if len(self.vertices) > 1:  # Ensure at least two vertices exist to draw a line
            GL.glUseProgram(self.shader.render_idx)
            self.vao.activate()
            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(self.vertices))