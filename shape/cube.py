from libs.shader import *
from libs import transform as T
from libs.buffer import *
from PIL import Image
import glm

class Cube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [-0.5, 0.5, 0.5],  #vertex 1
            [0.5, 0.5, 0.5],   #vertex 2
            [0.5, -0.5, 0.5],  #vertex 3
            [-0.5, -0.5, 0.5], #vertex 4
            [-0.5, 0.5, -0.5], #vertex 5
            [0.5, 0.5, -0.5],  #vertex 6
            [0.5, -0.5, -0.5], #vertex 7
            [-0.5, -0.5, -0.5] #vertex 8
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 3, 2,  # Front face (0-1-2-3)
            4, 5, 6, 7,  # Back face (4-5-6-7)
            0, 1, 4, 5,  # Bottom face (0-1-5-4)
            3, 2, 7, 6,  # Top face (3-2-6-7)
            0, 3, 4, 7,  # Left face (0-3-7-4)
            1, 2, 5, 6   # Right face (1-2-6-5)
        ], dtype=np.uint32)

        self.normals = np.array([
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],  # Front face
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],  # Back face
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],  # Top face
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],  # Bottom face
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],  # Left face
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]  # Right face
        ], dtype=np.float32)

        self.tex_coords = np.array([
            # Front face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 1, 2, 3, 4
            # Back face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 5, 6, 7, 8
            # Top face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 5, 6, 2, 1
            # Bottom face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 8, 7, 3, 4
            # Left face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 1, 4, 8, 5
            # Right face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]   # Vertex 2, 3, 7, 6
        ], dtype=np.float32)

        # colors: RGB format
        self.colors = np.array([
            [0.8, 0.2, 0.3],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #
     

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.tex_coords, ncomponents=2, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Load the texture
        self.uma.setup_texture("texture1", "texture/paper.jpg")  # Replace with your texture path

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(2, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)
        
        model_view_matrix = self.view * self.model
        
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
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
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

