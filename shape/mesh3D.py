from libs.shader import *
from libs.buffer import *
import glm
import numpy as np

class MathFunction:
    def __init__(self, vert_shader, frag_shader, func):
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        self.function = func
        self.generate_vertex_data()
        
        # Create a color gradient from blue to green
        y_values = self.vertices[:, 1]  # Get all y values
        self.y_min, self.y_max = np.min(y_values), np.max(y_values)
        
        # Generate colors for all vertices at once using vectorized operations
        self.colors = self.rgb_vectorized(self.y_min, self.y_max, y_values)

        # Create critical points and saddle points list
        self.critical_points = []
        self.saddle_points = []

    def rgb_vectorized(self, minimum, maximum, values):
        """Vectorized version of rgb function to handle arrays"""
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (values - minimum) / (maximum - minimum)
        
        # Apply a nonlinear transformation to the ratio to enhance contrast
        ratio = np.sin(ratio * np.pi / 2)  # Sine mapping for smoother gradients

        # Create empty array for colors
        colors = np.zeros((len(values), 3), dtype=np.float32)
        
        # Calculate RGB components using NumPy operations
        colors[:, 0] = ratio          # Red channel (increases from 0 to 1 as ratio goes from 0 to 1)
        colors[:, 2] = 1 - ratio
        
        return colors

    def evaluate_derivatives(self, x, z, func, delta = 1e-5):
        """
        Evaluate the first partial derivatives of a function at a point (x, z) using PyTorch autograd.

        :param x: The x-coordinate at which to evaluate the derivatives.
        :param z: The z-coordinate at which to evaluate the derivatives.
        :param func: The function for which to calculate the derivatives.
        :return: dy_dx, dy_dz
        """
        # Calculate partial derivative respect to x
        dy_dx = (func(x + delta, z) - func(x - delta, z)) / (2 * delta)

        # Calculate partial derivative respect to x
        dy_dz = (func(x, z + delta) - func(x, z - delta)) / (2 * delta)

        return dy_dx, dy_dz
    
    def generate_vertex_data(self):
        self.range_x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        self.range_z = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        self.resolution = len(self.range_x)

        X, Z = np.meshgrid(self.range_x, self.range_z)
        Y = self.function(X, Z)  # Use the provided function to compute Y
        vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        self.vertices = vertices.astype(np.float32)

        # Generate normals
        dx = np.diff(Y, axis=1, append=0)
        dz = np.diff(Y, axis=0, append=0)
        normals = np.dstack((-dx, np.ones_like(Y), -dz))
        normals /= np.linalg.norm(normals, axis=2)[:, :, np.newaxis]
        self.normals = normals.reshape(-1, 3).astype(np.float32)

        # Generate indices using triangle strips
        indices = []
        for row in range(self.resolution - 1):
            if row % 2 == 0:
                # Even row: process left to right
                for col in range(self.resolution - 1):
                    current_vertex = row * self.resolution + col
                    next_row_vertex = (row + 1) * self.resolution + col

                    # Add two triangles for the strip
                    indices.append(current_vertex)  # Current vertex
                    indices.append(next_row_vertex)  # Vertex in the row below
                    indices.append(current_vertex + 1)  # Next vertex in the row
                    indices.append(next_row_vertex + 1)  # Next vertex in the row below
            else:
                # Odd row: process right to left
                for col in range(self.resolution - 1, 0, -1):
                    current_vertex = row * self.resolution + col
                    next_row_vertex = (row + 1) * self.resolution + col

                    # Add two triangles for the strip
                    indices.append(current_vertex)  # Current vertex
                    indices.append(next_row_vertex)  # Vertex in the row below
                    indices.append(current_vertex - 1)  # Previous vertex in the row
                    indices.append(next_row_vertex - 1)  # Previous vertex in the row below
        
        self.indices = np.array(indices, dtype=np.uint32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 20), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(35.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model

        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Adjusted light position for the new view
        I_light = np.array([
            [0.7, 0.7, 0.7],  # diffuse
            [0.6, 0.6, 0.6],  # specular
            [0.2, 0.2, 0.2],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([8.0, -8.0, 15.0], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        K_materials = np.array([
            [0.6, 0.8, 0.6],  # diffuse
            [0.4, 0.6, 0.8],  # specular
            [0.2, 0.2, 0.3],  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 32.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')

        return self

    def draw(self):
        GL.glUseProgram(self.shader.render_idx)

        model_view_matrix = self.view  * self.model
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices), GL.GL_UNSIGNED_INT, None)

class Graph:
    def __init__(self, vert_shader, frag_shader, func):
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        self.function = func
        self.generate_vertex_data()
        
        # Create a color gradient from blue to green
        y_values = self.vertices[:, 1]  # Get all y values
        self.y_min, self.y_max = np.min(y_values), np.max(y_values)
        
        # Generate colors for all vertices at once using vectorized operations
        self.colors = self.rgb_vectorized(self.y_min, self.y_max, y_values)

        # Create critical points and saddle points list
        self.critical_points = []
        self.saddle_points = []

    def rgb_vectorized(self, minimum, maximum, values):
        """Vectorized version of rgb function to handle arrays"""
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (values - minimum) / (maximum - minimum)
        
        # Create empty array for colors
        colors = np.zeros((len(values), 3), dtype=np.float32)
        
        # Calculate RGB components using NumPy operations
        colors[:, 0] = ratio          # Red channel (increases from 0 to 1 as ratio goes from 0 to 1)
        colors[:, 2] = 1 - ratio
        
        return colors

    def evaluate_derivatives(self, x, z, func, delta = 1e-5):
        """
        Evaluate the first partial derivatives of a function at a point (x, z) using PyTorch autograd.

        :param x: The x-coordinate at which to evaluate the derivatives.
        :param z: The z-coordinate at which to evaluate the derivatives.
        :param func: The function for which to calculate the derivatives.
        :return: dy_dx, dy_dz
        """
        # Calculate partial derivative respect to x
        dy_dx = (func(x + delta, z) - func(x - delta, z)) / (2 * delta)

        # Calculate partial derivative respect to x
        dy_dz = (func(x, z + delta) - func(x, z - delta)) / (2 * delta)

        return dy_dx, dy_dz
    
    def generate_vertex_data(self):
        self.range_x = np.linspace(-3, 3, 200)
        self.range_z = np.linspace(-3, 3, 200)
        self.resolution = len(self.range_x)

        X, Z = np.meshgrid(self.range_x, self.range_z)
        Y = self.function(X, Z)  # Use the provided function to compute Y

        # Scale Y to get mesh look better
        self.Y_min, self.Y_max = Y.min(), Y.max()
        Y = 3 * (Y - self.Y_min) / (self.Y_max - self.Y_min) - 1.5  # Rescale to [-1.5, 1.5]
        vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        self.vertices = vertices.astype(np.float32)

        # Generate normals
        dx = np.diff(Y, axis=1, append=0)
        dz = np.diff(Y, axis=0, append=0)
        normals = np.dstack((-dx, np.ones_like(Y), -dz))
        normals /= np.linalg.norm(normals, axis=2)[:, :, np.newaxis]
        self.normals = normals.reshape(-1, 3).astype(np.float32)

        # Generate indices using triangle strips
        indices = []
        for row in range(self.resolution - 1):
            if row % 2 == 0:
                # Even row: process left to right
                for col in range(self.resolution - 1):
                    current_vertex = row * self.resolution + col
                    next_row_vertex = (row + 1) * self.resolution + col

                    # Add two triangles for the strip
                    indices.append(current_vertex)  # Current vertex
                    indices.append(next_row_vertex)  # Vertex in the row below
                    indices.append(current_vertex + 1)  # Next vertex in the row
                    indices.append(next_row_vertex + 1)  # Next vertex in the row below
            else:
                # Odd row: process right to left
                for col in range(self.resolution - 1, 0, -1):
                    current_vertex = row * self.resolution + col
                    next_row_vertex = (row + 1) * self.resolution + col

                    # Add two triangles for the strip
                    indices.append(current_vertex)  # Current vertex
                    indices.append(next_row_vertex)  # Vertex in the row below
                    indices.append(current_vertex - 1)  # Previous vertex in the row
                    indices.append(next_row_vertex - 1)  # Previous vertex in the row below
        
        self.indices = np.array(indices, dtype=np.uint32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 20), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(35.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model

        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Adjusted light position for the new view
        I_light = np.array([
            [0.7, 0.7, 0.7],  # diffuse
            [0.6, 0.6, 0.6],  # specular
            [0.2, 0.2, 0.2],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([8.0, -8.0, 15.0], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        K_materials = np.array([
            [0.6, 0.8, 0.6],  # diffuse
            [0.4, 0.6, 0.8],  # specular
            [0.2, 0.2, 0.3],  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 32.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')

        return self

    def draw(self):
        GL.glUseProgram(self.shader.render_idx)

        model_view_matrix = self.view  * self.model
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices), GL.GL_UNSIGNED_INT, None)