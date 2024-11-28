from libs.shader import *
from libs.buffer import *
from PIL import Image
import glm
import numpy as np
import math

class Sphere:
    def __init__(self, vert_shader, frag_shader, radius=0.05, sectors=20, stacks=20):
        self.radius = radius
        self.sectors = sectors
        self.stacks = stacks
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        self.generate_sphere()

    def generate_sphere(self):

        vertices = []
        normals = []
        indices = []

        sector_step = 2 * math.pi / self.sectors
        stack_step = math.pi / self.stacks

        for i in range(self.stacks + 1):
            stack_angle = math.pi / 2 - i * stack_step
            xy = self.radius * math.cos(stack_angle)
            z = self.radius * math.sin(stack_angle)

            for j in range(self.sectors + 1):
                sector_angle = j * sector_step

                x = xy * math.cos(sector_angle)
                y = xy * math.sin(sector_angle)

                vertices.append([x, y, z])
                normals.append([x/self.radius, y/self.radius, z/self.radius])

        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1

            for j in range(self.sectors):
                if i != 0:
                    indices.extend([k1, k2, k1 + 1])
                if i != (self.stacks - 1):
                    indices.extend([k1 + 1, k2, k2 + 1])

                k1 += 1
                k2 += 1

        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)

        # Generate random colors for each vertex
        self.colors = np.tile([1.0, 1.0, 0.0], (len(vertices), 1)).astype(np.float32) # yellow

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model 

        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Light setup (you can modify these values)
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials setup (you can modify these values)
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
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices), GL.GL_UNSIGNED_INT, None)

class SubdividedSphere:
    def __init__(self, vert_shader, frag_shader, radius=0.2, subdivisions=3):
        self.radius = radius
        self.subdivisions = subdivisions
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.vao = VAO()
        
        self.vertices = []
        self.indices = []
        self.normals = []
        self.colors = []
        self.tex_coords = []
        
        self.generate_sphere()

    def normalize_vertex(self, v):
        """Normalize vertex to lie on sphere surface"""
        length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return [v[0]/length * self.radius, 
                v[1]/length * self.radius, 
                v[2]/length * self.radius]

    def get_middle_point(self, p1, p2):
        """Find the middle point between two vertices and project to sphere"""
        middle = [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2, (p1[2] + p2[2])/2]
        return self.normalize_vertex(middle) # push midpoints outwards onto the surface of the sphere

    def generate_sphere(self):
        # Start with vertices of a regular tetrahedron
        t = (1.0 + math.sqrt(5.0)) / 2.0
        
        # Initial vertices of tetrahedron
        base_vertices = [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1]
        ]
        
        # Initial faces of icosahedron
        base_indices = [
            0, 11, 5,
            0, 5, 1,
            0, 1, 7,
            0, 7, 10,
            0, 10, 11,
            1, 5, 9,
            5, 11, 4,
            11, 10, 2,
            10, 7, 6,
            7, 1, 8,
            3, 9, 4,
            3, 4, 2,
            3, 2, 6,
            3, 6, 8,
            3, 8, 9,
            4, 9, 5,
            2, 4, 11,
            6, 2, 10,
            8, 6, 7,
            9, 8, 1
        ]

        # Normalize all vertices to lie on sphere
        vertices = [self.normalize_vertex(v) for v in base_vertices]
        
        # Perform subdivision
        for _ in range(self.subdivisions):
            new_indices = []
            index_map = {}
            
            for i in range(0, len(base_indices), 3):
                v1 = vertices[base_indices[i]]
                v2 = vertices[base_indices[i + 1]]
                v3 = vertices[base_indices[i + 2]]
                
                # Get or create middle points
                a = tuple(self.get_middle_point(v1, v2))
                b = tuple(self.get_middle_point(v2, v3))
                c = tuple(self.get_middle_point(v3, v1))
                
                # Add new vertices if they don't exist
                for v in [a, b, c]:
                    if v not in index_map:
                        index_map[v] = len(vertices)
                        vertices.append(list(v))
                
                # Create new triangles
                vi1, vi2, vi3 = base_indices[i:i+3]
                a_idx, b_idx, c_idx = index_map[a], index_map[b], index_map[c]
                
                new_indices.extend([
                    vi1, a_idx, c_idx,
                    vi2, b_idx, a_idx,
                    vi3, c_idx, b_idx,
                    a_idx, b_idx, c_idx
                ])
            
            base_indices = new_indices

        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(base_indices, dtype=np.uint32)
        
        # Generate normals (for a sphere, normals are just normalized vertices)
        self.normals = np.array([v/np.linalg.norm(v) for v in vertices], dtype=np.float32)
        
        # Generate random colors for vertices
        self.colors = np.tile([1.0, 1.0, 0.0], (len(vertices), 1)).astype(np.float32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Initial transformations
        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
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
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)