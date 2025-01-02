import OpenGL.GL as GL              
import glfw                         
import numpy as np 
import random
import re
import glm
from itertools import cycle
import imgui
from imgui.integrations.glfw import GlfwRenderer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import io
import copy

from libs.buffer import *
from libs.camera import *
from libs.shader import *
from libs.transform import *

from triangle import *
from rectangle import *
from tetrahedron import *
from pyramid import *
from cube import *
from cylinder import *
from sphere import *
from mesh3D import *
from model3D import *
from line import *

PYTHONPATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PYTHONPATH)

# ------------  Viewer class & windows management ------------------------------
class Viewer:
    """ GLFW viewer windows, with classic initialization & graphics loop """
    def __init__(self, width=1200, height=800):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        
        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # Initialize imgui
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)

        # Enable depth testing
        GL.glEnable(GL.GL_DEPTH_TEST)
        
        # Initialize shader
        self.gouraud_vert = './shader/gouraud.vert'
        self.gouraud_frag = './shader/gouraud.frag'
        self.phong_vert = "./shader/phong.vert"
        self.phong_frag = "./shader/phong.frag"
        self.phong_texture_vert = "./shader/phong_texture.vert"
        self.phong_texture_frag = "./shader/phong_texture.frag"
        self.phong_shader = Shader(self.phong_vert, self.phong_frag)
        self.flat_vert = "./shader/flat.vert"
        self.flat_frag = "./shader/flat.frag"

        # Initialize mouse parameters
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.left_mouse_pressed = False
        self.yaw = -90.0
        self.pitch = 0.0

        # Initialize camera parameters
        self.cameraSpeed = 0.5
        self.cameraPos = glm.vec3(0.0, 0.0, 5.0)   
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)  
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)    
        self.lastFrame = 0.0

        # Field of view for zooming
        self.fov = 45.0

        # Initialize option 
        self.trackball_option = False
        self.rotate_option = False
        self.triangle_option = False
        self.pyramid_option = False
        self.rectangle_option = False
        self.tetrahedron_option = False
        self.cube_option = False
        self.cylinder_option = False
        self.sphere_option = False
        self.subsphere_option = False
        self.mesh_option = False
        self.obj_option = False
        self.optimizer_option = False
        self.two_optimizers = False
        self.multi_camera_option = False
        self.move_camera_option = False

        # Variables for user selection
        self.select_optim = -1
        self.selected_shape = -1
        self.selected_obj = -1
        self.selected_mesh = -1
        self.selected_pyramid = None
        self.shape_created = False
        self.obj_file_path = None
        self.obj_file_name = "No file selected"
        self.optimization_method = ""

        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # Initialize model matrix
        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        # Initialize object class
        sphere =  None
        self.mathfunc = None # Control object MathFunction
        self.line = Line(self.gouraud_vert, self.gouraud_frag)

        # Initialize sphere parameters
        self.init_x = -0.41045934
        self.init_y = self.mathfunc(self.init_x, self.init_z) if self.mathfunc else 0.0
        self.init_z = 0.22101657

        self.obj1_center_x = -0.41045934
        self.obj1_center_y = self.mathfunc(self.obj1_center_x, self.obj1_center_z) if self.mathfunc else 0.0
        self.obj1_center_z = 0.22101657

        self.obj2_center_x = -0.41045934
        self.obj2_center_y = self.mathfunc(self.obj2_center_x, self.obj2_center_z) if self.mathfunc else 0.0
        self.obj2_center_z = 0.22101657

        # Ball parameters
        self.rotate_direction = 0.0
        self.rotation_speed = 2.0
        self.radius = 0.05
        
        self.prev1_x = 0.0
        self.prev1_y = 0.0
        self.prev1_z = 0.0
        self.prev1_angle = 0.0

        self.prev2_x = 0.0
        self.prev2_y = 0.0
        self.prev2_z = 0.0
        self.prev2_angle = 0.0
        
        # Initialize Adam optimizer parameters
        self.m_x, self.m_z = 0.0, 0.0  # First moment vectors
        self.v_x, self.v_z = 0.0, 0.0  # Second moment vectors
        self.beta1, self.beta2 = 0.9, 0.999  # Decay rates for moments
        self.epsilon = 1e-8  # Small constant to prevent division by zero
        self.t = 1  # Timestep
        
        # Initial learning rate and position initialized
        self.random_point = None
        self.position1_initialized = True
        self.position2_initialized = True
        self.learning_rate = 0.0

        # Initialize math function 
        self.values = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        self.selected_function = ''
        self.func = None # get function representation

        # Initialize control flags
        self.move_left_flag = True
        self.rotate_changed = False
        self.radius_changed = False
        self.lr_changed = False

        # Initialize camera
        self.camera = Camera()

        # Initialize previous time
        self.prev_time = 0.0

        # Initialize matplotlib figure for contour plot
        self.fig = Figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111)
        self.contour_texture = None
        self.show_contour = True

        # Initialize path trail
        self.ball1_trail = []
        self.ball2_trail = []

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        
        # Use trackball
        # glfw.set_mouse_button_callback(self.win, self.click_choose_pyramid)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.scroll_callback)

        # glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_DISABLED)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        # GL.glClearColor(0.0, 0.0, 0.0, 1.0) # Black background

        # initially empty list of object to draw
        self.drawables = []

    def imgui_menu(self):
        imgui.new_frame()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(300, 200)
        imgui.begin("UI", True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

        # Add rotation speed slider
        imgui.set_next_item_width(100)
        imgui.text("Learning Rate: %.03f" % self.learning_rate)
        
        imgui.same_line()
        if imgui.button("-"):
            self.learning_rate = max(self.learning_rate - 0.001, 0.0)  # Prevent going below min_value

        imgui.same_line()
        if imgui.button("+"):
            self.learning_rate = min(self.learning_rate + 0.001, 0.1)  # Prevent exceeding max_value

        # Add radius slider
        imgui.set_next_item_width(100)
        self.radius_changed, radius_value = imgui.slider_float("Radius", 
                                          self.radius, 
                                          min_value=0.05, 
                                          max_value=2,
                                          format="%.2f")
        if self.radius_changed:
            self.radius = radius_value

        # Existing shape selection code
        imgui.set_next_item_width(100)
        if imgui.begin_combo("Select Shape", "Shapes"):
            # Add checkboxes inside the combo
            _, self.triangle_option = imgui.checkbox("Triangle", self.triangle_option)
            _, self.rectangle_option = imgui.checkbox("Rectangle", self.rectangle_option)
            _, self.tetrahedron_option = imgui.checkbox("Tetrahedron", self.tetrahedron_option)
            _, self.pyramid_option = imgui.checkbox("Pyramid", self.pyramid_option)
            _, self.cube_option = imgui.checkbox("Cube", self.cube_option)
            _, self.cylinder_option = imgui.checkbox("Cylinder", self.cylinder_option)
            _, self.sphere_option = imgui.checkbox("Sphere", self.sphere_option)
            _, self.subsphere_option = imgui.checkbox("Subdivided Sphere", self.subsphere_option)
            _, self.mesh_option = imgui.checkbox("Mesh", self.mesh_option)
            _, self.obj_option = imgui.checkbox("Object", self.obj_option)
            
            imgui.end_combo()

        # Existing shape selection code
        imgui.set_next_item_width(100)
        if imgui.begin_combo("Select Option", "Options"):
            # Add checkboxes inside the combo
            _, self.rotate_option = imgui.checkbox("Rotate Object", self.rotate_option)
            _, self.trackball_option = imgui.checkbox("Use Trackball", self.trackball_option)
            _, self.optimizer_option = imgui.checkbox("Optimizer", self.optimizer_option)
            _, self.two_optimizers = imgui.checkbox("Visualize 2 Optimizers", self.two_optimizers)
            _, self.multi_camera_option = imgui.checkbox("Multi Camera", self.multi_camera_option)
            _, self.move_camera_option = imgui.checkbox("Move Camera", self.move_camera_option)
            imgui.end_combo()

        if self.optimizer_option:
            
            imgui.set_next_window_position(0, 250)  
            imgui.set_next_window_size(300, 100) 

            # Start new frame
            imgui.begin("Optimizer", True)
            imgui.set_next_item_width(100)
            
            imgui.set_next_item_width(100)
            _, self.select_optim = imgui.combo(
                "Select Optimizer",
                self.select_optim,
                [
                    "SGD",
                    "Adam"
                ]
            )
            optim_list = ['SGD', 'Adam']
            self.optimization_method = optim_list[self.select_optim]

            if imgui.button("Random Position", width=120):
                self.random_position()
            imgui.same_line()
            if imgui.button("Restart", width=100):
                self.restart()
            
            imgui.end()
        
        if self.two_optimizers:
            
            imgui.set_next_window_position(0, 250)  
            imgui.set_next_window_size(300, 100) 

            # Start new frame
            imgui.begin("Optimizer", True)
            imgui.set_next_item_width(100)

            if imgui.button("Random Position", width=120):
                self.random_position()
            imgui.same_line()
            if imgui.button("Restart", width=100):
                self.restart()
            
            imgui.end()

        # If DynamicMesh is selected, show function selection
        if self.mesh_option:
            imgui.set_next_item_width(100)
            _, self.selected_mesh = imgui.combo(
                "Select Mesh",
                self.selected_mesh,
                [ "3*(1-x)**2*exp(-x**2-(z+1)**2)-10*(x/5 - x**3 - z**5)*exp(-x**2-z**2) - 1/3*exp(-(x+1)**2-z**2)",
                "sin(sqrt(x**2+z**2))",
                "(x-1)**2+(z-1)**2",
                "sin(x)+cos(z)",
                "sin(x)"]
            )
            selected_formula = [
                "3*(1-x)**2*exp(-x**2-(z+1)**2)-10*(x/5 - x**3 - z**5)*exp(-x**2-z**2) - 1/3*exp(-(x+1)**2-z**2)",
                "sin(sqrt(x**2+z**2))",
                "(x-1)**2+(z-1)**2",
                "sin(x)+cos(z)",
                "sin(x)"
            ][self.selected_mesh]
            self.selected_function = selected_formula

            # Add checkbox for contour plot
            imgui.set_next_item_width(100)
            _, self.show_contour = imgui.checkbox("Show Contour", self.show_contour)

        # If OBJ File is selected, show file selection button and file name
        if self.obj_option:
            imgui.set_next_item_width(100)
            _, self.selected_obj = imgui.combo(
                "Select Obj",
                self.selected_obj,
                ["Wuson",
                 "Porsche",
                 "Rubik"]
            )

        # Confirm button
        if imgui.button("Confirm", width=100):

            # Enable to draw multi-cam
            self.create_model()
            # Reset model rotation when creating new shape
            self.model = glm.mat4(1.0)

        # Create contour plot
        self.render_contour_plot()

        imgui.end()
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def random_position(self):
        self.position1_initialized = True
        self.position2_initialized = True
        self.random_point = random.choice(self.mathfunc.vertices)
        self.obj1_center_x, self.obj1_center_y, self.obj1_center_z = self.random_point
        self.obj2_center_x, self.obj2_center_y, self.obj2_center_z = self.random_point
        self.init_x, self.init_y, self.init_z = self.random_point
        
        # Reset 2d trail on contour map
        self.ball1_trail = []
        self.ball2_trail = []

        # Reset 3d trail on 3d mesh
        self.line.reset_vertex()

    def restart(self):
        self.position1_initialized = True
        self.position2_initialized = True
        self.obj1_center_x, self.obj1_center_y, self.obj1_center_z = self.init_x, self.init_y, self.init_z
        self.obj2_center_x, self.obj2_center_y, self.obj2_center_z = self.init_x, self.init_y, self.init_z

        # Reset 2d trail on contour map
        self.ball1_trail = []
        self.ball2_trail = []

        # Reset 3d trail on 3d mesh
        self.line.reset_vertex()

    def update_contour_trail(self):
        if self.learning_rate == 0:
            return
        if hasattr(self, 'obj1_center_x') and hasattr(self, 'obj1_center_z'):
            # Append the current position as a tuple (x, z)
            self.ball1_trail.append((self.obj1_center_x, self.obj1_center_z))

        if hasattr(self, 'obj2_center_x') and hasattr(self, 'obj2_center_z'):
            # Append the current position as a tuple (x, z)
            self.ball2_trail.append((self.obj2_center_x, self.obj2_center_z))

    def render_contour_trail(self):
        if not self.ball1_trail and not self.ball2_trail:
            return
        
        for x, z in self.ball1_trail:
            self.ax.plot(x, z, 'yo', markersize=1)  # Plot yellow dots on the trail

        if self.two_optimizers:
            for x, z in self.ball2_trail:
                self.ax.plot(x, z, 'go', markersize=1)  # Plot green dots on the trail

    def update_3d_trail(self):
        self.line.model = glm.mat4(1.0)
        self.line.view = self.trackball.view_matrix2(self.cameraPos)
        self.line.projection = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)
        self.line.add_vertex((self.obj1_center_x, self.obj1_center_y, self.obj1_center_z))
        self.line.draw()

    def setup_framebuffers(self, num_cameras):
        """Initialize framebuffers and textures for each camera"""
        for i in range(num_cameras):
            # Generate and bind framebuffer
            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            
            # Generate texture for color attachment
            texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 
                           self.width, self.height, 0,
                           GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            
            # Attach texture to framebuffer
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                     GL.GL_TEXTURE_2D, texture, 0)
            
            # Generate and attach depth renderbuffer
            rbo = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rbo)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24,
                                   self.width, self.height)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                       GL.GL_RENDERBUFFER, rbo)
            
            # Check framebuffer completeness
            if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
                print(f"Framebuffer {i} is not complete!")
            
            self.framebuffers.append(fbo)
            self.textures.append(texture)
            self.depth_renderbuffers.append(rbo)
            
        # Reset to default framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def update_contour_plot(self):
        if not self.show_contour:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        self.X, self.Z = np.meshgrid(self.mathfunc.range_x, self.mathfunc.range_x)

        # Get the current function if it exists
        if hasattr(self, 'func') and self.func is not None:
            # Calculate Y values for contour
            Y = self.func(self.X, self.Z)
            
            # Create filled contour plot with hot-cold color scheme
            contour = self.ax.contourf(self.X, self.Z, Y, levels=20, 
                                     cmap='coolwarm')  # Changed to coolwarm colormap
            
            # Plot first ball position
            if hasattr(self, 'obj1_center_x') and hasattr(self, 'obj1_center_z'):
                self.ax.plot(self.obj1_center_x, self.obj1_center_z, 'ko',  # Black outline
                           markerfacecolor='y',  # White fill
                           markersize=10, 
                           label='Ball Position')
                
                # Add height value annotation near the ball
                height = self.func(self.obj1_center_x, self.obj1_center_z) + self.radius
                self.ax.annotate(f'Height: {height:.2f}', 
                               (self.obj1_center_x, self.obj1_center_z),
                               xytext=(10, 10), textcoords='offset points')
            
            if self.two_optimizers:
                # Plot second ball position
                if hasattr(self, 'obj2_center_x') and hasattr(self, 'obj2_center_z'):
                    self.ax.plot(self.obj2_center_x, self.obj2_center_z, 'ko',  # Black outline
                            markerfacecolor='g',  # White fill
                            markersize=10, 
                            label='Ball Position')
                    
                    # Add height value annotation near the ball
                    height = self.func(self.obj2_center_x, self.obj2_center_z) + self.radius
                    self.ax.annotate(f'Height: {height:.2f}', 
                                (self.obj2_center_x, self.obj2_center_z),
                                xytext=(10, 10), textcoords='offset points')

            self.ax.set_title('Surface Height Contour Map')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Z Position')
            self.ax.set_ylim(self.Z.max(), self.Z.min())
            self.ax.grid(True, linestyle='--', alpha=0.3)
            
            # Create path trail
            self.render_contour_trail()

            # Add legend
            self.ax.legend()
            
            # Update plot layout
            self.fig.tight_layout()
            
            # Convert plot to texture
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            
            # Get the RGBA buffer from the figure
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (h, w, 4)
            
            # Flip ARGB to RGBA
            buf = np.roll(buf, 3, axis=2)
            
            # Convert to OpenGL texture
            if self.contour_texture is None:
                self.contour_texture = GL.glGenTextures(1)
            
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.contour_texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                           GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buf)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    def render_contour_plot(self):
        if not self.show_contour or self.contour_texture is None:
            return
            
        # Set up ImGui window for contour plot
        imgui.set_next_window_position(900, 0, imgui.ALWAYS)  # Changed to COND_ONCE
        imgui.set_next_window_size(300, 300)  
        expanded, visible = imgui.begin("Height Contour Map", True)
        
        if expanded:
            # Get the window draw list for custom rendering
            draw_list = imgui.get_window_draw_list()
            
            # Get current window position and size
            win_pos = imgui.get_cursor_screen_pos()
            content_width = imgui.get_window_width() - 20  # Padding
            content_height = imgui.get_window_height() - 40  # Space for title
            
            # Bind and draw texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.contour_texture)
            draw_list.add_image(
                self.contour_texture,
                (win_pos.x, win_pos.y),
                (win_pos.x + content_width, win_pos.y + content_height)
            )

        imgui.end()

    def update_pyramid_view(self):
        if self.selected_pyramid:
            # Render the view of the selected pyramid in a small window
            viewport_size = (300, 200)
            GL.glViewport(0, 0, *viewport_size)

            # Clear the color and depth buffers
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Render the scene from the selected pyramid's perspective
            self.selected_pyramid.draw()

            # Reset the viewport to the full window size
            GL.glViewport(0, 0, *self.win_size)

    def update_radius_sphere(self):
        for drawable in self.drawables:
            if isinstance(drawable, (Sphere, SubdividedSphere)) and self.radius_changed:
                drawable.radius = self.radius
                drawable.generate_sphere()
                drawable.setup()
                # drawable.draw()

    def rotate(self):
        for drawable in self.drawables:
            time = glfw.get_time()
            drawable.model = glm.rotate(glm.mat4(1.0), glm.radians(time * self.rotation_speed), glm.vec3(0.0, 1.0, 0.0))
            drawable.view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
            drawable.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

    def SGD_visualization(self, sphere):
        view_matrix = glm.lookAt(glm.vec3(0, 10, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        projection_matrix = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)

        self.mathfunc.model = glm.mat4(1.0)
        self.mathfunc.view = view_matrix
        self.mathfunc.projection = projection_matrix
        
        sphere.view = view_matrix
        sphere.projection = projection_matrix

        # Get the normal vector from your mathematical function
        dx, dz = self.mathfunc.evaluate_derivatives(self.obj1_center_x, self.obj1_center_z, self.mathfunc.function)
        normal_vector = glm.normalize(glm.vec3(-dx, 1, -dz))

        # Update the position of ball on the surface at first
        if self.position1_initialized:
            self.obj1_center_x += self.radius * normal_vector.x
            self.obj1_center_y += self.radius * normal_vector.y
            self.obj1_center_z += self.radius * normal_vector.z
            self.position1_initialized = False

        # Calculate movement and normal vectors as before
        translation_vector = glm.vec3(self.obj1_center_x, self.obj1_center_y, self.obj1_center_z)
        movement_vector = glm.vec3(self.obj1_center_x, self.obj1_center_y, self.obj1_center_z) - glm.vec3(self.prev1_x, self.prev1_y, self.prev1_z)

        # Calculate rotation based on movement
        center_angle, rotation_axis = self.calculate_rotation(movement_vector, self.radius, normal_vector)

        # Accumulate rotation
        self.prev1_angle += center_angle

        # Create rotation matrix around the correct axis
        rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-self.prev1_angle), rotation_axis)

        # Combine translation and rotation
        # First translate to position, then apply rotation
        if movement_vector != glm.vec3(0.0, 0.0, 0.0):
            sphere.model = glm.translate(glm.mat4(1.0), translation_vector) * rotation_matrix
        else:
            sphere.model = glm.translate(glm.mat4(1.0), translation_vector) 

        # Update positions as before
        self.prev1_x = self.obj1_center_x
        self.prev1_y = self.obj1_center_y
        self.prev1_z = self.obj1_center_z

        # Update new position
        if not (dx < np.exp(-8) and dz < np.exp(-8)):
            self.obj1_center_x = self.obj1_center_x - self.learning_rate * dx
            self.obj1_center_z = self.obj1_center_z - self.learning_rate * dz
            y = self.mathfunc.function(self.obj1_center_x, self.obj1_center_z)
            self.obj1_center_y = 2 * (y - self.mathfunc.Y_min) / (self.mathfunc.Y_max - self.mathfunc.Y_min) - 1 # scale to range [-1, 1]
            self.obj1_center_y += self.radius # to lay sphere on surface

        # Draw path trail of the ball
        self.update_3d_trail()

    def Adam_visualization(self, sphere):
        view_matrix = glm.lookAt(glm.vec3(0, 10, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        projection_matrix = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)

        self.mathfunc.model = glm.mat4(1.0)
        self.mathfunc.view = view_matrix
        self.mathfunc.projection = projection_matrix

        sphere.view = view_matrix
        sphere.projection = projection_matrix

        # Get the gradient of the surface
        dx, dz = self.mathfunc.evaluate_derivatives(self.obj2_center_x, self.obj2_center_z, self.mathfunc.function)
        normal_vector = glm.normalize(glm.vec3(-dx, 1, -dz))

        # Update the position of the ball on the surface initially
        if self.position2_initialized:
            self.obj2_center_x += self.radius * normal_vector.x
            self.obj2_center_y += self.radius * normal_vector.y
            self.obj2_center_z += self.radius * normal_vector.z
            self.position2_initialized = False

        # Adam optimizer step
        g_x, g_z = dx, dz  # Gradients
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * g_x
        self.m_z = self.beta1 * self.m_z + (1 - self.beta1) * g_z
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * (g_x ** 2)
        self.v_z = self.beta2 * self.v_z + (1 - self.beta2) * (g_z ** 2)

        # Bias-corrected moments
        m_x_hat = self.m_x / (1 - self.beta1 ** self.t)
        m_z_hat = self.m_z / (1 - self.beta1 ** self.t)
        v_x_hat = self.v_x / (1 - self.beta2 ** self.t)
        v_z_hat = self.v_z / (1 - self.beta2 ** self.t)

        # Update positions using Adam formula
        self.obj2_center_x -= self.learning_rate * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
        self.obj2_center_z -= self.learning_rate * m_z_hat / (np.sqrt(v_z_hat) + self.epsilon)

        # Compute updated Y position
        y = self.mathfunc.function(self.obj2_center_x, self.obj2_center_z)
        self.obj2_center_y = 2 * (y - self.mathfunc.Y_min) / (self.mathfunc.Y_max - self.mathfunc.Y_min) - 1  # Scale to range [-1, 1]
        self.obj2_center_y += self.radius  # Lay sphere on surface

        # Increment timestep
        self.t += 1

        # Calculate movement and normal vectors as before
        translation_vector = glm.vec3(self.obj2_center_x, self.obj2_center_y, self.obj2_center_z)
        movement_vector = glm.vec3(self.obj2_center_x, self.obj2_center_y, self.obj2_center_z) - glm.vec3(self.prev2_x, self.prev2_y, self.prev2_z)

        # Calculate rotation based on movement
        center_angle, rotation_axis = self.calculate_rotation(movement_vector, self.radius, normal_vector)

        # Accumulate rotation
        self.prev2_angle += center_angle

        # Create rotation matrix around the correct axis
        rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-self.prev2_angle), rotation_axis)

        # Combine translation and rotation
        # First translate to position, then apply rotation
        if movement_vector != glm.vec3(0.0, 0.0, 0.0):
            sphere.model = glm.translate(glm.mat4(1.0), translation_vector) * rotation_matrix
        else:
            sphere.model = glm.translate(glm.mat4(1.0), translation_vector)

        # Update previous positions
        self.prev2_x = self.obj2_center_x
        self.prev2_y = self.obj2_center_y
        self.prev2_z = self.obj2_center_z

    def visualize_2_optimizers(self):
        # Check if the drawables is empty or not
        if len(self.drawables) < 2: # At least have 2 spheres
            return
        
        spheres = []
        for drawble in self.drawables:
            if isinstance(drawble, (Sphere, SubdividedSphere)):
                spheres.append(drawble)

        # Set up for sphere 1
        sphere1 = spheres[0]
        sphere1.colors = np.tile([1.0, 1.0, 0.0], (len(sphere1.vertices), 1)).astype(np.float32) # yellow
        sphere1.setup()
        self.SGD_visualization(sphere1)

        # Set up for sphere 2
        sphere2 = spheres[1]
        sphere2.colors = np.tile([0.0, 1.0, 0.0], (len(sphere2.vertices), 1)).astype(np.float32) # green
        sphere2.setup()
        self.Adam_visualization(sphere2)


    def move_camera_around(self):
        for drawable in self.drawables:
            drawable.model = glm.mat4(1.0)
            drawable.view = self.trackball.view_matrix2(self.cameraPos)
            drawable.projection = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)
    
    def use_trackball(self):
        for drawable in self.drawables:
                win_size = glfw.get_window_size(self.win)
                drawable.view = self.trackball.view_matrix3()
                drawable.projection = self.trackball.projection_matrix(win_size)

    def multi_cam(self):
        # Set up some parameters
        rows = 3
        cols = 3
        left_width = 1200
        left_height = 800
        right_width = 1200
        right_height = 800
        cell_width = right_width // cols
        cell_height = right_height // rows

        # Define the hemisphere of multi-camera
        sphere = Sphere(self.phong_vert, self.phong_frag).setup()
        sphere.radius = 4.0
        sphere.generate_sphere()

        self.pyramids = []
        self.num_pyramids = 9
        for i in range(self.num_pyramids):
            pyramid = Pyramid(self.phong_shader)
            pyramid.setup()
            
            # Position pyramid around the object
            theta = i * (2 * np.pi) / self.num_pyramids  # Distribute evenly around the sphere
            x = sphere.radius * np.cos(theta)
            z = sphere.radius * np.sin(theta)

            P = glm.vec3(x,0,z)
            O = glm.vec3(0.0, 0.0, 0.0)
            OP = O - P # direction vector of pyramid

            # First, move the pyramid to the position in sphere
            translation = glm.translate(glm.mat4(1.0), P) 
            
            # Split direction vector into 2 component vector
            y_component_vec = glm.vec3(0.0, OP.y, 0.0) 
            xz_component_vec = glm.vec3(OP.x, 0.0, OP.z)
            
            # Two steps to set the direction of the camera directly look into the sphere center
            # Step 1: rotate around y-axis
            init_vec1 = glm.vec3(0.0,0.0,-1.0)
            dot_product = glm.dot(xz_component_vec, init_vec1)
            magnitude_xz_component_vec = glm.length(xz_component_vec)
            magnitude_init_vec1 = glm.length(init_vec1)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_xz_component_vec * magnitude_init_vec1)

            # Ensure the cosine is within the valid range for arccos due to floating point precision
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Calculate the angle in radians
            y_angle = np.arccos(cos_theta)

            if P.x < 0: # If position if left-half sphere
                y_axis_rotation = glm.rotate(glm.mat4(1.0), -y_angle, glm.vec3(0.0, 1.0, 0.0))
            else: # If position if left-half sphere
                y_axis_rotation = glm.rotate(glm.mat4(1.0), y_angle, glm.vec3(0.0, 1.0, 0.0))

            # Step 2: rotate around z-axis
            init_vec2 = glm.vec3(1.0,0.0,0.0)
            dot_product = glm.dot(OP, init_vec2)
            magnitude_OP = glm.length(OP)
            magnitude_init_vec2 = glm.length(init_vec2)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_OP * magnitude_init_vec2)

            # Ensure the cosine is within the valid range for arccos due to floating point precision
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Calculate the angle in radians
            z_angle = np.arccos(cos_theta)
            z_axis_rotation = glm.rotate(glm.mat4(1.0), glm.radians(z_angle), glm.vec3(0.0, 0.0, 1.0))

            # Apply model matrix for pyramid
            pyramid.model = translation * y_axis_rotation 
            
            # Set up view matrix for camera
            eye = P
            at = glm.vec3(0,0,0)
            up = glm.normalize(glm.vec3(pyramid.model[1]))
            
            pyramid.view = glm.lookAt(eye, at, up)
            pyramid.projection = glm.perspective(glm.radians(self.fov), cell_width / cell_height, 0.1, 100.0)
            
            self.pyramids.append(pyramid)

        # Show multi-cam system on the left
        GL.glViewport(0, 0, left_width, left_height)
        self.new_pyramids = [pyramid.clone() for pyramid in self.pyramids]
        for i in range(len(self.pyramids)):  
            # self.new_pyramids[i].setup()
            self.new_pyramids[i].model = self.pyramids[i].model
            # self.new_pyramids[i].view = self.pyramids[i].view
            # self.new_pyramids[i].projection = self.pyramids[i].projection
            self.new_pyramids[i].draw()

        for drawable in self.drawables:
            win_size = glfw.get_window_size(self.win)
            drawable.view = self.trackball.view_matrix3()
            drawable.projection = self.trackball.projection_matrix(win_size)
            drawable.draw()

        # Render each camera's view on the right
        for i, pyramid in enumerate(self.pyramids):
            row, col = divmod(i, cols)
            GL.glViewport(left_width + col * cell_width, row * cell_height, cell_width, cell_height)
            for drawable in self.drawables:
                drawable.view = pyramid.view
                drawable.projection = pyramid.projection
                drawable.model = self.trackball.matrix()
                drawable.draw()

    def run(self):
        """ Main render loop for this OpenGL windows """
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffer

            # Update radius for sphere
            self.update_radius_sphere()

            if self.rotate_option:
                ########################################################################
                #                         Rotate the object                            #
                ########################################################################
                self.rotate()
            
            if self.optimization_method == "SGD":
                ########################################################################
                #                           SGD Visualization                          #
                ########################################################################
                sphere = None
                for drawable in self.drawables:
                    if isinstance(drawable, (MathFunction, Graph)):
                        self.mathfunc = drawable

                    if isinstance(drawable, (Sphere, SubdividedSphere)):
                        sphere = drawable
                self.SGD_visualization(sphere)

            if self.optimization_method == "Adam":
                ########################################################################
                #                           Adam Visualization                         #
                ########################################################################
                sphere = None
                for drawable in self.drawables:
                    if isinstance(drawable, (MathFunction, Graph)):
                        self.mathfunc = drawable

                    if isinstance(drawable, (Sphere, SubdividedSphere)):
                        sphere = drawable
                self.Adam_visualization(sphere)

            if self.two_optimizers:
                ########################################################################
                #                     Two Optimizers Visualization                     #
                ########################################################################
                for drawable in self.drawables:
                    if isinstance(drawable, (MathFunction, Graph)):
                        self.mathfunc = drawable
                self.visualize_2_optimizers()

            if self.multi_camera_option:
                ########################################################################
                #                             Multi-camera                             #
                ########################################################################
                self.multi_cam()

            if self.move_camera_option:
                ########################################################################
                #                          Move camera around                          #
                ########################################################################
                self.move_camera_around()
            
            if self.trackball_option:
                ########################################################################
                #                   Rotate and Zoom using Trackball                    #
                ########################################################################
                self.use_trackball()
            
            if not self.optimizer_option and not self.move_camera_option and not self.trackball_option and not self.multi_camera_option:
                for drawable in self.drawables:
                    drawable.model = glm.mat4(1.0)
                    drawable.view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
                    drawable.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

            if not self.multi_camera_option:
                for drawable in self.drawables:
                    drawable.draw()
            
            # Update path trail
            self.update_contour_trail()

            # Update and render contour plot
            if self.mathfunc:
                self.update_contour_plot()

            self.update_pyramid_view()

            # Render GUI menu
            self.imgui_menu()

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()
            self.imgui_impl.process_inputs()

        self.imgui_impl.shutdown()

    def add(self, drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_J:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            if key == glfw.KEY_W:
                self.cameraPos += self.cameraSpeed * self.cameraFront
            if key == glfw.KEY_S:
                self.cameraPos -= self.cameraSpeed * self.cameraFront
            if key == glfw.KEY_A:
                self.cameraPos -= glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed
            if key == glfw.KEY_D:
                self.cameraPos += glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed
            
            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    # def click_choose_pyramid(self, window, button, action, mods):
    #     if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
    #         # Perform ray casting to select a pyramid
    #         mouse_pos = glfw.get_cursor_pos(self.win)
    #         ray_origin, ray_direction = self.camera.get_ray(mouse_pos)
    #         for pyramid in self.pyramids:
    #             if pyramid.intersects_ray(ray_origin, ray_direction):
    #                 self.selected_pyramid = pyramid
    #                 break
    #         else:
    #             self.selected_pyramid = None

    #     # Call the original mouse button callback
    #     super().click_choose_pyramid(window, button, action, mods)

    def on_mouse_move(self, window, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(window)[1] - ypos)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(window))

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)
    
    def scroll_callback(self, window, xoffset, yoffset):
        self.fov -= float(yoffset)
        # if self.fov < 1.0:
        #     self.fov = 1.0
        # if self.fov > 45.0:
        #     self.fov = 45.0
        self.trackball.zoom(yoffset, glfw.get_window_size(window)[1])

    def create_model(self):        
        model = []

        # Clear previous shapes before creating a new one
        self.drawables.clear()
        
        # Create shape based on user's choice
        if self.triangle_option:  # Triangle
            model.append(Triangle(self.phong_vert, self.phong_frag).setup())
        if self.rectangle_option:  # Rectangle
            model.append(Rectangle(self.phong_vert, self.phong_frag).setup())
        if self.tetrahedron_option:  # TetraHedron
            model.append(TetraHedron(self.phong_texture_vert, self.phong_texture_frag).setup())
        if self.pyramid_option:  # pyramid
            model.append(Pyramid(self.phong_shader).setup())
        if self.cube_option: # Cube
            model.append(Cube(self.phong_texture_vert, self.phong_texture_frag).setup())
        if self.cylinder_option: # Cylinder
            model.append(Cylinder(self.phong_vert, self.phong_frag).setup())
        if self.sphere_option: # Sphere
            model.append(Sphere(self.gouraud_vert, self.gouraud_frag).setup())
        if self.subsphere_option: # Sphere
            model.append(SubdividedSphere(self.phong_vert, self.phong_frag).setup())    
        if self.mesh_option and self.selected_function:  # DynamicMesh
            self.func = get_function(self.selected_function)
            if self.selected_function == '3*(1-x)**2*exp(-x**2-(z+1)**2)-10*(x/5 - x**3 - z**5)*exp(-x**2-z**2) - 1/3*exp(-(x+1)**2-z**2)':
                model.append(Graph("shader/phong.vert", "shader/phong.frag", self.func).setup())    
            else:
                model.append(MathFunction("shader/phong.vert", "shader/phong.frag", self.func).setup())
        if self.obj_option:  # OBJ File
            if self.selected_obj == 0:
                chosen_obj = 'obj/WusonOBJ.obj'
            elif self.selected_obj == 1:
                chosen_obj = 'obj/Porsche_911_GT2.obj'
            elif self.selected_obj == 2:
                chosen_obj = 'obj/rubik.obj'
            model.append(Obj("shader/phong.vert", "shader/phong.frag", chosen_obj).setup())
        if self.two_optimizers:
            sphere1 = Sphere(self.phong_vert, self.phong_frag).setup()
            sphere2 = Sphere(self.phong_vert, self.phong_frag).setup()
            model.extend([sphere1, sphere2])
        if self.multi_camera_option:
            model.extend(self.pyramids)
            model.append(Cylinder(self.phong_vert, self.phong_frag).setup())

        # Add the created model to the viewer's drawables and mark as created
        if model:
            self.add(model)
            self.shape_created = True

    def calculate_rotation(self, movement_vector, radius, normal_vector):
        # Get the distance moved
        distance = glm.length(movement_vector)
        
        if distance < 0.0001:  # Prevent division by zero and tiny rotations
            return 0.0, glm.vec3(1, 0, 0)
        
        # Calculate rotation angle based on arc length
        # The sphere should rotate by (distance/radius) radians
        theta = distance / radius
        angle_degrees = math.degrees(theta)
        
        # Calculate rotation axis
        # The rotation axis should be perpendicular to both movement direction and normal
        movement_dir = glm.normalize(movement_vector)
        rotation_axis = glm.cross(movement_dir, normal_vector)
        
        # If rotation axis is zero (movement parallel to normal), use a default axis
        if glm.length(rotation_axis) < 0.0001:
            rotation_axis = glm.vec3(1, 0, 0)
        else:
            rotation_axis = glm.normalize(rotation_axis)
        
        return angle_degrees, rotation_axis

def get_function(input):
    function_map = {
        "sin": "np.sin",
        "cos": "np.cos",
        "tan": "np.tan",
        "exp": "np.exp",
        "log": "np.log",
        "sqrt": "np.sqrt",
        "abs": "np.abs"
    }

    for func, np_func in function_map.items():
        input = re.sub(rf"\b{func}\b", np_func, input)

    def function_representation(X, Z):
        try:
            return eval(input, {"__builtins__": None}, {"np": np, "x": X, "z": Z})
        except Exception as e:
            print(f"Error evaluating function: {e}")
            return None  
    return function_representation

# -------------- main program and scene setup --------------------------------
def main():
    """ create windows, add shaders & scene objects, then run rendering loop """
    viewer = Viewer()
    viewer.run()

if __name__ == '__main__':
    glfw.init()                # initialize windows system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts