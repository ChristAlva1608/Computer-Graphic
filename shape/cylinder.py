from libs.shader import *
from libs.buffer import *
import glm
import numpy as np
import math
import ctypes

class Cylinder:
    def __init__(self, vert_shader, frag_shader, radius=1.0, height=2.0, sectors=32):
        self.radius = radius
        self.height = height
        self.sectors = sectors
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()
        
        self.generate_cylinder()

    def generate_cylinder(self):
        vertices = []
        normals = []
        indices = []
        colors = []
        
        # Generate vertices for sides (using triangle strip)
        sector_step = 2 * math.pi / self.sectors
        
        # Side vertices (2 circles of vertices for top and bottom)
        for i in range(self.sectors + 1):
            sector_angle = i * sector_step
            
            x = self.radius * math.cos(sector_angle)
            y = self.radius * math.sin(sector_angle)
            
            # Bottom vertex
            vertices.append([x, y, -self.height/2])
            normals.append([x/self.radius, y/self.radius, 0.0])
            colors.append(np.random.rand(3))
            
            # Top vertex
            vertices.append([x, y, self.height/2])
            normals.append([x/self.radius, y/self.radius, 0.0])
            colors.append(np.random.rand(3))
        
        # Indices for side (triangle strip)
        for i in range(2 * (self.sectors + 1)):
            indices.append(i)
            
        # Store the number of indices for the side
        self.side_indices_count = len(indices)
        
        # Store starting vertex index for caps
        start_cap_vertex = len(vertices)
        
        # Generate vertices for top and bottom faces (using triangle fan)
        # Center points
        vertices.append([0, 0, self.height/2])  # Top center
        normals.append([0, 0, 1])
        colors.append(np.random.rand(3))
        
        # Top circle points
        for i in range(self.sectors + 1):
            sector_angle = i * sector_step
            x = self.radius * math.cos(sector_angle)
            y = self.radius * math.sin(sector_angle)
            vertices.append([x, y, self.height/2])
            normals.append([0, 0, 1])
            colors.append(np.random.rand(3))
            
        vertices.append([0, 0, -self.height/2])  # Bottom center
        normals.append([0, 0, -1])
        colors.append(np.random.rand(3))
        
        # Bottom circle points
        for i in range(self.sectors + 1):
            sector_angle = i * sector_step
            x = self.radius * math.cos(sector_angle)
            y = self.radius * math.sin(sector_angle)
            vertices.append([x, y, -self.height/2])
            normals.append([0, 0, -1])
            colors.append(np.random.rand(3))
        
        # Store indices for top face (triangle fan)
        self.top_start_index = len(indices)
        center_top = start_cap_vertex  # Index of top center
        for i in range(self.sectors + 1):
            indices.append(center_top)  # Center point
            indices.append(center_top + 1 + i)  # Points around the circle
            if i < self.sectors:
                indices.append(center_top + 1 + (i + 1))  # Next point
        
        # Store indices for bottom face (triangle fan)
        self.bottom_start_index = len(indices)
        center_bottom = start_cap_vertex + self.sectors + 2  # Index of bottom center
        for i in range(self.sectors + 1):
            indices.append(center_bottom)  # Center point
            indices.append(center_bottom + 1 + i)  # Points around the circle
            if i < self.sectors:
                indices.append(center_bottom + 1 + (i + 1))  # Next point
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.colors = np.array(colors, dtype=np.float32)
        
        # Calculate the number of indices for caps
        self.top_indices_count = 3 * self.sectors  # 3 vertices per triangle * sectors
        self.bottom_indices_count = 3 * self.sectors  # 3 vertices per triangle * sectors

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Initial transformations
        self.model = glm.mat4(1.0)
        self.view = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, -5.0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model

        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Light setup
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials setup
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7],  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')

        return self

    def draw(self):
        GL.glUseProgram(self.shader.render_idx)
        
        model_view_matrix = self.view * self.model
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        
        # Draw cylinder side using triangle strip
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.side_indices_count, GL.GL_UNSIGNED_INT, None)
        
        # Draw top face using triangle fan
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, self.top_indices_count, GL.GL_UNSIGNED_INT, 
                         ctypes.c_void_p(self.top_start_index * 4))
        
        # Draw bottom face using triangle fan
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, self.bottom_indices_count, GL.GL_UNSIGNED_INT,
                         ctypes.c_void_p(self.bottom_start_index * 4))