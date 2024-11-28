import glfw
from OpenGL.GL import glGetString, GL_VERSION

def check_opengl_version():
    # Initialize glfw
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    
    # Set the required OpenGL version (optional)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "OpenGL Version Check", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Get OpenGL version
    version = glGetString(GL_VERSION)
    print("OpenGL Version:", version.decode())

    # Clean up and terminate
    glfw.destroy_window(window)
    glfw.terminate()

check_opengl_version()
