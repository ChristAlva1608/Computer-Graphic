from libs.shader import *
from libs.buffer import *
import glm
import numpy as np

class PathTrail:
    def __init__(self, max_points=1000):
        self.max_points = max_points
        self.positions = []
        
        # Shader for the path
        self.vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """
        
        self.fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        void main() {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color for the path
        }
        """
        
        # Create and compile shaders
        self.shader = compileProgram(
            compileShader(self.vertex_shader, GL_VERTEX_SHADER),
            compileShader(self.fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        # Initialize VAO and VBO
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        # Set up vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        # Get uniform locations
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")

    def add_position(self, pos):
        """Add a new position to the path trail"""
        self.positions.append(pos)
        if len(self.positions) > self.max_points:
            self.positions.pop(0)
            
        # Update VBO with new positions
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        positions_array = np.array(self.positions, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, positions_array.nbytes, positions_array, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, view_matrix, projection_matrix):
        """Draw the path trail"""
        if len(self.positions) < 2:
            return
            
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, glm.value_ptr(view_matrix))
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, glm.value_ptr(projection_matrix))
        
        # Draw the path
        glBindVertexArray(self.vao)
        glLineWidth(2.0)  # Set line width
        glDrawArrays(GL_LINE_STRIP, 0, len(self.positions))
        glBindVertexArray(0)
        
        glUseProgram(0)

    def cleanup(self):
        """Clean up OpenGL resources"""
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.shader)