import vulkan as vk
import glfw
WIDTH = 800
HEIGHT = 800
glfw.init()
window = glfw.create_window(640, 480, "Test Vulkan", None, None)
glfw.make_context_current(window)
extensionCount=0
extensions = vk.vkEnumerateInstanceExtensionProperties(None)
for e in extensions:
 extensionCount=extensionCount+1
print('availables extensions:', extensionCount)
# Loop until the user closes the window
while not glfw.window_should_close(window):
 # Render here, e.g. using pyOpenGL
 # Swap front and back buffers
 glfw.swap_buffers(window)
 # Poll for and process events
 glfw.poll_events()
glfw.terminate()
