from libs.shader import *
from libs import transform as T
from libs.buffer import *
from PIL import Image
import glm

def load_texture(image_paths):
    # Load multiple 2D images and stack them to create a 3D texture
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.convert('RGB')
        images.append(image)

    # Get dimensions from the first image
    width, height = images[0].size
    
    # Resize all images to the same size (based on the first image)
    for i in range(len(images)):
        images[i] = images[i].resize((width, height))

    # Create an array to hold the pixel data for the 3D texture
    image_data = np.zeros((len(images), height, width, 3), dtype=np.uint8)
    
    # Populate the array with pixel data from each image
    for i, image in enumerate(images):
        image_data[i] = np.array(image)

    # Generate a 3D texture ID
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, texture_id)

    # Set texture parameters
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    # Upload the texture data to OpenGL
    GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB, width, height, len(images), 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image_data)

    # Generate mipmaps
    GL.glGenerateMipmap(GL.GL_TEXTURE_3D)

    return texture_id


class TetraHedron(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [0.5, 0.5, 0.5],    # Vertex 1
            [-0.5, -0.5, 0.5],  # Vertex 2
            [-0.5, 0.5, -0.5],  # Vertex 3
            [0.5, -0.5, -0.5],  # Vertex 4
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2,  # Face 1
            0, 1, 3,  # Face 2
            0, 2, 3,  # Face 3
            1, 2, 3   # Face 4
        ], dtype=np.uint32)

        self.normals = np.array([
            [0.0, 0.577, 0.577],   # Normal for Face 1
            [0.577, 0.0, 0.577],   # Normal for Face 2
            [0.577, 0.577, 0.0],   # Normal for Face 3
            [0.0, -0.577, -0.577], # Normal for Face 4
        ], dtype=np.float32)

        self.tex_coords = np.array([
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],  # Texture coordinates for Face 1
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],  # Texture coordinates for Face 2
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],  # Texture coordinates for Face 3
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],  # Texture coordinates for Face 4
        ], dtype=np.float32)

        # colors: RGB format
        self.colors = np.array([
            [1.0, 0.0, 0.0],  # Color for Vertex 1
            [0.0, 1.0, 0.0],  # Color for Vertex 2
            [0.0, 0.0, 1.0],  # Color for Vertex 3
            [1.0, 1.0, 0.0],  # Color for Vertex 4
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
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.tex_coords, ncomponents=2, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Set up flag useTexCoord
        useTexCoordLocation = GL.glGetUniformLocation(self.shader.render_idx, "useTexCoord");
        if hasattr(self, 'tex_coords') and self.tex_coords.size > 0:
            GL.glUniform1i(useTexCoordLocation, 1)
        else:
            GL.glUniform1i(useTexCoordLocation, 0)

        # Load the texture
        self.texture_id = load_texture(["texture/paper.jpg","texture/sky.jpeg","texture/sand.jpg"])  # Replace with your texture path

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
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

         # Bind the texture
        GL.glActiveTexture(GL.GL_TEXTURE0)  
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.texture_id)

        model_view_matrix = self.view * self.model
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

