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

from libs.buffer import *
from libs.camera import *
from libs.shader import *
from libs.transform import *

from triangle import *
from rectangle import *
from tetrahedron import *
from shape.pyramid import *
from cube import *
from cylinder import *
from sphere import *
from mesh3D import *
from model3D import *

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
        self.phong_vert = "./shader/phong.vert"
        self.phong_frag = "./shader/phong.frag"
        self.phong_vert1 = "./shader/phong1.vert"
        self.phong_frag1 = "./shader/phong1.frag"
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
        self.combo_open = False
        self.trackball_option = False
        self.rotate_option = False
        self.triangle_option = False
        self.frustum_option = False
        self.rectangle_option = False
        self.tetrahedron_option = False
        self.cube_option = False
        self.cylinder_option = False
        self.sphere_option = False
        self.subsphere_option = False
        self.mesh_option = False
        self.obj_option = False
        self.gradient_option = False
        self.multi_camera_option = False
        self.move_camera_option = False
        self.random_position = False

        # Variables for user selection
        self.selected_shape = -1
        self.selected_obj = -1
        self.selected_mesh = -1
        self.selected_frustum = None
        self.shape_created = False
        self.obj_file_path = None
        self.obj_file_name = "No file selected"

        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # Initialize model matrix
        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        # Initialize sphere parameters
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_z = 0.0
        self.obj_center_x = 0.0
        self.obj_center_y = 0.0
        self.obj_center_z = 0.0

        self.rotate_direction = 0.0
        self.rotation_speed = 2.0
        self.radius = 0.1
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0
        self.prev_angle = 0.0
        
        # Initial learning rate 
        self.learning_rate = 0.0

        # Initialize math function 
        self.values = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        self.selected_function = ''
        self.func = None

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

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        
        # Use trackball
        # glfw.set_mouse_button_callback(self.win, self.click_choose_frustum)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

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
        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        imgui.set_next_window_size(300, 200)
        imgui.begin("UI", True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

        # Add rotation speed slider
        imgui.set_next_item_width(100)
        imgui.text("Learning Rate: %.03f" % self.learning_rate)
        
        imgui.same_line()
        if imgui.button("-"):
            self.learning_rate = max(self.learning_rate - 0.001, 0.001)  # Prevent going below min_value

        imgui.same_line()
        if imgui.button("+"):
            self.learning_rate = min(self.learning_rate + 0.001, 0.1)  # Prevent exceeding max_value

        # Add radius slider
        imgui.set_next_item_width(100)
        self.radius_changed, radius_value = imgui.slider_float("Radius", 
                                          self.radius, 
                                          min_value=0.1, 
                                          max_value=2,
                                          format="%.1f")
        if self.radius_changed:
            self.radius = radius_value

        # Existing shape selection code
        imgui.set_next_item_width(100)
        if imgui.begin_combo("Select Shape", "Shapes"):
            # Add checkboxes inside the combo
            _, self.triangle_option = imgui.checkbox("Triangle", self.triangle_option)
            _, self.rectangle_option = imgui.checkbox("Rectangle", self.rectangle_option)
            _, self.tetrahedron_option = imgui.checkbox("Tetrahedron", self.tetrahedron_option)
            _, self.frustum_option = imgui.checkbox("Frustum", self.frustum_option)
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
            _, self.gradient_option = imgui.checkbox("Gradient Descent", self.gradient_option)
            _, self.multi_camera_option = imgui.checkbox("Multi Camera", self.multi_camera_option)
            _, self.move_camera_option = imgui.checkbox("Move Camera", self.move_camera_option)
            imgui.end_combo()

        if self.gradient_option:
            imgui.set_next_item_width(100)
            if imgui.button("Random Position", width=120):
                self.drawables.clear()
                self.random_position = True

        # If DynamicMesh is selected, show function selection
        if self.mesh_option:
            imgui.set_next_item_width(100)
            _, self.selected_mesh = imgui.combo(
                "Select Mesh",
                self.selected_mesh,
                ["sin(sqrt(x**2+z**2))",
                 "(x-1)**2+(z-1)**2",
                 "sin(x)+cos(z)",
                 "sin(x)"]
            )
            selected_formula = [
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

    def update_contour_plot(self):
        if not self.show_contour:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        self.X, self.Z = np.meshgrid(self.values, self.values)
        # Get the current function if it exists
        if hasattr(self, 'func') and self.func is not None:
            # Calculate Y values for contour
            Y = self.func(self.X, self.Z)
            
            # Create filled contour plot with hot-cold color scheme
            contour = self.ax.contourf(self.X, self.Z, Y, levels=20, 
                                     cmap='coolwarm')  # Changed to coolwarm colormap
            
            # Plot current ball position
            if hasattr(self, 'obj_center_x') and hasattr(self, 'obj_center_z'):
                self.ax.plot(self.obj_center_x, self.obj_center_z, 'ko',  # Black outline
                           markerfacecolor='w',  # White fill
                           markersize=10, 
                           label='Ball Position')
                
                # Add height value annotation near the ball
                height = self.func(self.obj_center_x, self.obj_center_z) + self.radius
                self.ax.annotate(f'Height: {height:.2f}', 
                               (self.obj_center_x, self.obj_center_z),
                               xytext=(10, 10), textcoords='offset points')
            
            self.ax.set_title('Surface Height Contour Map')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Z Position')
            self.ax.grid(True, linestyle='--', alpha=0.3)
            
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

    def update_frustum_view(self):
        if self.selected_frustum:
            # Render the view of the selected frustum in a small window
            viewport_size = (300, 200)
            GL.glViewport(0, 0, *viewport_size)

            # Clear the color and depth buffers
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Render the scene from the selected frustum's perspective
            self.selected_frustum.draw()

            # Reset the viewport to the full window size
            GL.glViewport(0, 0, *self.win_size)

    def run(self):
        """ Main render loop for this OpenGL windows """
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffer

            # Update radius for sphere
            for drawable in self.drawables:
                if isinstance(drawable, (Sphere, SubdividedSphere)) and self.radius_changed:
                    drawable.radius = self.radius
                    drawable.generate_sphere()
                    drawable.setup()
                    # drawable.draw()

            if self.rotate_option:
                ########################################################################
                #                         Rotate the object                            #
                ########################################################################
                for drawable in self.drawables:
                    time = glfw.get_time()
                    drawable.model = glm.rotate(glm.mat4(1.0), glm.radians(time * self.rotation_speed), glm.vec3(0.0, 1.0, 0.0))
                    drawable.view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
                    drawable.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

            if self.gradient_option:
                ########################################################################
                #                    Gradient Descent Visualization                    #
                ########################################################################
                self.func = get_function("sin(sqrt(x**2+z**2))")
                mathfunc = MathFunction(self.phong_vert, self.phong_frag, self.func).setup()
                sphere =  Sphere(self.phong_vert, self.phong_frag).setup()

                if self.random_position:
                    self.obj_center_x, self.obj_center_y, self.obj_center_z = random.choice(mathfunc.vertices)
                    self.obj_center_y += self.radius
                    # Assign the (x, y, z) components of the random vertex
                    # self.obj_center_x = float(random.choice(self.values))
                    # self.obj_center_z = float(random.choice(self.values))
                    # self.obj_center_y = mathfunc.function(self.obj_center_x, self.obj_center_z) + self.radius  # Adjust y to lay the sphere on the surface

                    print('x: ', self.obj_center_x)
                    print('z: ', self.obj_center_z)
                    print('viewer y: ', self.obj_center_y)
                    # sphere.model =  glm.translate(glm.mat4(1.0), glm.vec3(self.obj_center_x, self.obj_center_y, self.obj_center_z)) 
                    self.random_position = False

                view_matrix = glm.lookAt(glm.vec3(0, 10, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
                projection_matrix = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)

                mathfunc.model = glm.mat4(1.0)
                mathfunc.view = view_matrix
                mathfunc.projection = projection_matrix
                
                sphere.view = view_matrix
                sphere.projection = projection_matrix

                translation_vector = glm.vec3(self.obj_center_x, self.obj_center_y, self.obj_center_z)
                movement_vector = glm.vec3(self.obj_center_x, self.obj_center_y, self.obj_center_z) - glm.vec3(self.prev_x, self.prev_y, self.prev_z)
                
                # Calculate normal vector at a position
                dx, dz = mathfunc.evaluate_derivatives(self.obj_center_x, self.obj_center_z, mathfunc.function)
                # normal_vector = glm.vec3(-dx,1,-dz)
                normal_vector = glm.vec3(0,1,0)

                center_angle, rotation_axis = self.calculate_rotation(movement_vector, self.radius, normal_vector)
                self.prev_angle += center_angle
                self.rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-self.prev_angle), rotation_axis)
                
                sphere.model = glm.translate(glm.mat4(1.0), translation_vector) * self.rotation_matrix
                # sphere.model = glm.translate(glm.mat4(1.0), translation_vector) 

                # Update previous position
                self.prev_x = self.obj_center_x
                self.prev_y = self.obj_center_y
                self.prev_z = self.obj_center_z

                # Update new position
                self.obj_center_x = self.obj_center_x - self.learning_rate * dx
                self.obj_center_z = self.obj_center_z - self.learning_rate * dz
                self.obj_center_y = mathfunc.function(self.obj_center_x, self.obj_center_z) + self.radius

                self.drawables.extend([mathfunc, sphere])


            if self.multi_camera_option:
                ########################################################################
                #                             Multi-camera                             #
                ########################################################################
                # Draw the object in the center
                # self.add([Cylinder(self.phong_vert1, self.phong_frag1).setup()])

                # Define the hemisphere of multi-camera
                sphere = Sphere(self.phong_vert, self.phong_frag).setup()
                sphere.radius = 4.0
                sphere.generate_sphere()

                # self.frustums = []
                # self.num_frustums = 2
                # for i in range(self.num_frustums):
                #     frustum = Frustum(self.phong_vert, self.phong_frag)
                #     frustum.setup()
                    
                #     # Position frustum around the object
                #     theta = i * 2 * np.pi / self.num_frustums
                #     x = sphere.radius * np.cos(theta)
                #     z = sphere.radius * np.sin(theta)
                #     frustum.model = glm.translate(glm.mat4(1.0), glm.vec3(x, 0, z))
                #     self.frustums.append(frustum)

                for i in range(0,len(sphere.vertices),9):
                    P = glm.vec3(*sphere.vertices[i])
                    O = glm.vec3(0.0, 0.0, 0.0)
                    OP = O - P # direction vector of frustum

                    # First, move the frustum to the position in sphere
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
                    y_axis_rotation = glm.rotate(glm.mat4(1.0), glm.radians(y_angle), glm.vec3(0.0, 1.0, 0.0))

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

                    # Apply model matrix for frustum
                    frustum = Frustum(self.flat_vert, self.flat_frag).setup()
                    # frustum = Frustum(self.phong_vert, self.phong_frag).setup()
                    frustum.model = translation * y_axis_rotation * z_axis_rotation
                    # frustum.model = translation  * y_axis_rotation
                    
                    self.drawables.append(frustum)

                    # Allow selecting the frustum
                    frustum.selectable = True

            if self.move_camera_option:
                ########################################################################
                #                       Walk around the object                         #
                ########################################################################
                for drawable in self.drawables:
                    currentFrame = glfw.get_time()
                    deltaTime = currentFrame - self.lastFrame
                    drawable.lastFrame = currentFrame
                    drawable.cameraSpeed = 50 * deltaTime
                    
                    drawable.model = glm.mat4(1.0)
                    drawable.view = self.trackball.view_matrix2(self.cameraPos)
                    drawable.projection = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)
            
            if self.trackball_option:
                ########################################################################
                #                   Rotate and Zoom using Trackball                    #
                ########################################################################
                for drawable in self.drawables:
                    win_size = glfw.get_window_size(self.win)
                    drawable.view = self.trackball.view_matrix3()
                    drawable.projection = self.trackball.projection_matrix(win_size)
            
            if not self.gradient_option and not self.move_camera_option and not self.trackball_option and not self.multi_camera_option:
                for drawable in self.drawables:
                    drawable.model = glm.mat4(1.0)
                    drawable.view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
                    drawable.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

            for drawable in self.drawables:
                drawable.draw()
            
            # Update and render contour plot
            self.update_contour_plot()

            self.update_frustum_view()

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

    # def click_choose_frustum(self, window, button, action, mods):
    #     if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
    #         # Perform ray casting to select a frustum
    #         mouse_pos = glfw.get_cursor_pos(self.win)
    #         ray_origin, ray_direction = self.camera.get_ray(mouse_pos)
    #         for frustum in self.frustums:
    #             if frustum.intersects_ray(ray_origin, ray_direction):
    #                 self.selected_frustum = frustum
    #                 break
    #         else:
    #             self.selected_frustum = None

    #     # Call the original mouse button callback
    #     super().click_choose_frustum(window, button, action, mods)

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
        if self.fov < 1.0:
            self.fov = 1.0
        if self.fov > 45.0:
            self.fov = 45.0

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])

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
            model.append(TetraHedron(self.phong_vert1, self.phong_frag1).setup())
        if self.frustum_option:  # Frustum
            model.append(Frustum(self.phong_vert, self.phong_frag).setup())
        if self.cube_option: # Cube
            model.append(Cube(self.phong_vert1, self.phong_frag1).setup())
        if self.cylinder_option: # Cylinder
            model.append(Cylinder(self.phong_vert, self.phong_frag).setup())
        if self.sphere_option: # Sphere
            model.append(Sphere(self.phong_vert, self.phong_frag).setup())
        if self.subsphere_option: # Sphere
            model.append(SubdividedSphere(self.phong_vert, self.phong_frag).setup())    
        if self.mesh_option and self.selected_function:  # DynamicMesh
            self.func = get_function(self.selected_function)
            model.append(MathFunction("shader/phong.vert", "shader/phong.frag", self.func).setup())
        if self.obj_option:  # OBJ File
            if self.selected_obj == 0:
                chosen_obj = 'obj/WusonOBJ.obj'
            elif self.selected_obj == 1:
                chosen_obj = 'obj/Porsche_911_GT2.obj'
            elif self.selected_obj == 2:
                chosen_obj = 'obj/rubik.obj'
            model.append(Obj("shader/phong.vert", "shader/phong.frag", chosen_obj).setup())

        # Add the created model to the viewer's drawables and mark as created
        if model:
            self.add(model)
            self.shape_created = True

    def calculate_rotation(self, movement_vector, radius, up_vector=glm.vec3(0, 1, 0)):
        distance = glm.length(movement_vector)
        theta = distance / radius
        angle_degrees = math.degrees(theta)
        rotation_axis = glm.cross(movement_vector, up_vector)
        if glm.length(rotation_axis) != 0:
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