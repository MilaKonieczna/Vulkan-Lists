import os
from vulkan import *
import numpy as np
import glfw
import time
from PIL import Image
import tinyobjloader as tol
import glm

WIDTH = 800
HEIGHT = 800

validationLayers = vkEnumerateInstanceLayerProperties()
validationLayers = [l.layerName for l in validationLayers]
print("availables layers: %s\n" % validationLayers)

validationLayers = ["VK_LAYER_KHRONOS_validation"]
deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

enableValidationLayers = True


def debugCallback(*args):
    print('DEBUG: {} {}'.format(args[5], args[6]))
    return 0


def createDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkCreateDebugReportCallbackEXT')
    if func:
        return func(instance, pCreateInfo, pAllocator)
    else:
        return VK_ERROR_EXTENSION_NOT_PRESENT


def destroyDebugReportCallbackEXT(instance, callback, pAllocator):
    func = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
    if func:
        func(instance, callback, pAllocator)


def destroySurface(instance, surface, pAllocator=None):
    func = vkGetInstanceProcAddr(instance, 'vkDestroySurfaceKHR')
    if func:
        func(instance, surface, pAllocator)


def destroySwapChain(device, swapChain, pAllocator=None):
    func = vkGetDeviceProcAddr(device, 'vkDestroySwapchainKHR')
    if func:
        func(device, swapChain, pAllocator)


class QueueFamilyIndices(object):

    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


class SwapChainSupportDetails(object):
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None


class Vertex(object):
    POS = np.array([0, 0, 0], np.float32)
    COLOR = np.array([0, 0, 0], np.float32)
    TEXCOORD = np.array([0, 0], np.float32)

    @staticmethod
    def getBindingDescription():
        bindingDescription = VkVertexInputBindingDescription(
            binding=0,
            stride=Vertex.POS.nbytes + Vertex.COLOR.nbytes + Vertex.TEXCOORD.nbytes,
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )

        return bindingDescription

    @staticmethod
    def getAttributeDescriptions():
        pos = VkVertexInputAttributeDescription(
            location=0,
            binding=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=0
        )

        color = VkVertexInputAttributeDescription(
            location=1,
            binding=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=Vertex.POS.nbytes,
        )

        texcoord = VkVertexInputAttributeDescription(
            location=2,
            binding=0,
            format=VK_FORMAT_R32G32_SFLOAT,
            offset=Vertex.POS.nbytes + Vertex.COLOR.nbytes,
        )
        return [pos, color, texcoord]


class UniformBufferObject(object):

    def __init__(self):
        self.model = np.identity(4, np.float32)
        self.view = np.identity(4, np.float32)
        self.proj = np.identity(4, np.float32)

    def toArray(self):
        return np.concatenate((self.model, self.view, self.proj))

    @property
    def nbytes(self):
        return self.proj.nbytes + self.view.nbytes + self.model.nbytes


class HelloTriangleApplication(object):

    def __init__(self):
        self.__window = None
        self.__instance = None
        self.__callback = None
        self.__surface = None
        self.__physicalDevice = None
        self.__device = None
        self.__graphicsQueue = None
        self.__presentQueue = None

        self.__swapChain = None
        self.__swapChainImages = None
        self.__swapChainImageFormat = None
        self.__swapChainExtent = None

        self.__swapChainImageViews = None
        self.__swapChainFramebuffers = None

        self.__renderPass = None
        self.__pipelineLayout = None
        self.__graphicsPipeline = None

        self.__commandPool = None
        self.__commandBuffers = None

        self.__imageAvailableSemaphore = None
        self.__renderFinishedSemaphore = None

        self.__textureImage = None
        self.__textureImageMemory = None
        self.__textureImageView = None
        self.__textureSampler = None

        self.__depthImage = None
        self.__depthImageMemory = None
        self.__depthImageView = None

        self.__vertexBuffer = None
        self.__vertexBufferMemory = None

        self.__indexBuffer = None
        self.__indexBufferMemory = None

        self.__descriptorPool = None
        self.__descriptorSet = None
        self.__descriptorSetLayout = None
        self.__uniformBuffer = None
        self.__uniformBufferMemory = None

        self.modelPath = 'models/apple.obj'
        self.texturePath = 'textures/texture2.png'

        self.__vertices = []
        self.__indices = []
        self.__ubo = UniformBufferObject()

        """
        self.__vertices = np.array([
            # pos    color     texCoord
            -.5, -.5, 0, 1, 0, 0, 1, 0,
            .5, -.5, 0, 0, 1, 0, 0, 0,
            .5, .5, 0, 0, 0, 1, 0, 1,
            -.5, .5, 0, 1, 1, 1, 1, 1,
            -.5, -.5, -.5, 1, 0, 0, 1, 0,
            .5, -.5, -.5, 0, 1, 0, 0, 0,
            .5, .5, -.5, 0, 0, 1, 0, 1,
            -.5, .5, -.5, 1, 1, 1, 1, 1
        ], np.float32)

        self.__indices = np.array([0, 1, 2, 2, 3, 0,
                                   4, 5, 6, 6, 7, 4], np.uint16)

        self.__ubo = UniformBufferObject()

        """

        self.__startTime = time.time()

    def __del__(self):
        vkDeviceWaitIdle(self.__device)

        if self.__textureSampler:
            vkDestroySampler(self.__device, self.__textureSampler, None)

        if self.__textureImageView:
            vkDestroyImageView(self.__device, self.__textureImageView, None)

        if self.__textureImage:
            vkDestroyImage(self.__device, self.__textureImage, None)

        if self.__textureImageMemory:
            vkFreeMemory(self.__device, self.__textureImageMemory, None)

        if self.__descriptorPool:
            vkDestroyDescriptorPool(self.__device, self.__descriptorPool, None)

        if self.__uniformBuffer:
            vkDestroyBuffer(self.__device, self.__uniformBuffer, None)

        if self.__uniformBufferMemory:
            vkFreeMemory(self.__device, self.__uniformBufferMemory, None)

        if self.__indexBuffer:
            vkDestroyBuffer(self.__device, self.__indexBuffer, None)

        if self.__indexBufferMemory:
            vkFreeMemory(self.__device, self.__indexBufferMemory, None)

        if self.__vertexBuffer:
            vkDestroyBuffer(self.__device, self.__vertexBuffer, None)

        if self.__vertexBufferMemory:
            vkFreeMemory(self.__device, self.__vertexBufferMemory, None)

        if self.__imageAvailableSemaphore:
            vkDestroySemaphore(self.__device, self.__imageAvailableSemaphore, None)

        if self.__renderFinishedSemaphore:
            vkDestroySemaphore(self.__device, self.__renderFinishedSemaphore, None)

        if self.__descriptorSetLayout:
            vkDestroyDescriptorSetLayout(self.__device, self.__descriptorSetLayout, None)

        if self.__commandBuffers:
            self.__commandBuffers = None

        if self.__commandPool:
            vkDestroyCommandPool(self.__device, self.__commandPool, None)

        vkDestroyImageView(self.__device, self.__depthImageView, None)
        vkDestroyImage(self.__device, self.__depthImage, None)
        vkFreeMemory(self.__device, self.__depthImageMemory, None)

        if self.__swapChainFramebuffers:
            for i in self.__swapChainFramebuffers:
                vkDestroyFramebuffer(self.__device, i, None)
            self.__swapChainFramebuffers = None

        if self.__renderPass:
            vkDestroyRenderPass(self.__device, self.__renderPass, None)

        if self.__pipelineLayout:
            vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, None)

        if self.__graphicsPipeline:
            vkDestroyPipeline(self.__device, self.__graphicsPipeline, None)

        if self.__swapChainImageViews:
            for i in self.__swapChainImageViews:
                vkDestroyImageView(self.__device, i, None)

        if self.__swapChain:
            destroySwapChain(self.__device, self.__swapChain, None)

        if self.__device:
            vkDestroyDevice(self.__device, None)

        if self.__surface:
            destroySurface(self.__instance, self.__surface, None)

        if self.__callback:
            destroyDebugReportCallbackEXT(self.__instance, self.__callback, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)

    def __initWindow(self):
        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.__window = glfw.create_window(WIDTH, HEIGHT, "Vulkan", None, None)

    def __cleanupSwapChain(self):
        vkDestroyImageView(self.__device, self.__depthImageView, None)
        vkDestroyImage(self.__device, self.__depthImage, None)
        vkFreeMemory(self.__device, self.__depthImageMemory, None)

        [vkDestroyFramebuffer(self.__device, i, None) for i in self.__swapChainFramebuffers]
        self.__swapChainFramebuffers = []

        vkFreeCommandBuffers(self.__device, self.__commandPool, len(self.__commandBuffers), self.__commandBuffers)
        self.__swapChainFramebuffers = []

        vkDestroyPipeline(self.__device, self.__graphicsPipeline, None)
        vkDestroyPipelineLayout(self.__device, self.__pipelineLayout, None)
        vkDestroyRenderPass(self.__device, self.__renderPass, None)

        [vkDestroyImageView(self.__device, i, None) for i in self.__swapChainImageViews]
        self.__swapChainImageViews = []
        # vkDestroySwapchainKHR(self.__device, self.__swapChain, None)

    def __recreateSwapChain(self):
        vkDeviceWaitIdle(self.__device)

        self.__cleanupSwapChain()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createGraphicsPipeline()
        self.__createDepthResources()
        self.__createFramebuffers()
        self.__createCommandBuffers()

    def __initVulkan(self):
        self.__createInstance()
        self.__setupDebugCallback()
        self.__createSurface()
        self.__pickPhysicalDevice()
        self.__createLogicalDevice()
        self.__createSwapChain()
        self.__createImageViews()
        self.__createRenderPass()
        self.__createDescriptorSetLayout()
        self.__createGraphicsPipeline()
        self.__createCommandPool()
        self.__createDepthResources()
        self.__createFramebuffers()
        self.__createTextureImage()
        self.__createTextureImageView()
        self.__createTextureSampler()
        self.__loadModel()
        self.__createVertexBuffer()
        self.__createIndexBuffer()
        self.__createUniformBuffer()
        self.__createDescriptorPool()
        self.__createDescriptorSet()
        self.__createCommandBuffers()
        self.__createSemaphores()

    def __mainLoop(self):
        while not glfw.window_should_close(self.__window):
            glfw.poll_events()
            self.__drawFrame()
            vkDeviceWaitIdle(self.__device)

    def __createInstance(self):
        if enableValidationLayers and not self.__checkValidationLayerSupport():
            raise Exception("validation layers requested, but not available!")

        appInfo = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Hello Triangle',
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName='No Engine',
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 0, 3)
        )

        createInfo = None
        extensions = self.__getRequiredExtensions()

        if enableValidationLayers:
            createInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=appInfo,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers
            )
        else:
            createInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=appInfo,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                enabledLayerCount=0
            )

        self.__instance = vkCreateInstance(createInfo, None)

    def __setupDebugCallback(self):
        if not enableValidationLayers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback
        )
        self.__callback = createDebugReportCallbackEXT(self.__instance, createInfo, None)
        if not self.__callback:
            raise Exception("failed to set up debug callback!")

    def __createSurface(self):
        surface_ptr = ffi.new('VkSurfaceKHR[1]')
        glfw.create_window_surface(self.__instance, self.__window, None, surface_ptr)
        self.__surface = surface_ptr[0]
        if self.__surface is None:
            raise Exception("failed to create window surface!")

    def __pickPhysicalDevice(self):
        devices = vkEnumeratePhysicalDevices(self.__instance)

        for device in devices:
            if self.__isDeviceSuitable(device):
                self.__physicalDevice = device
                break

        if self.__physicalDevice is None:
            raise Exception("failed to find a suitable GPU!")

    def __createLogicalDevice(self):
        indices = self.__findQueueFamilies(self.__physicalDevice)
        uniqueQueueFamilies = {}.fromkeys((indices.graphicsFamily, indices.presentFamily))
        queueCreateInfos = []
        for queueFamily in uniqueQueueFamilies:
            queueCreateInfo = VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queueFamily,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            queueCreateInfos.append(queueCreateInfo)

        deviceFeatures = VkPhysicalDeviceFeatures()
        deviceFeatures.samplerAnisotropy = True
        createInfo = None
        if enableValidationLayers:
            createInfo = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                flags=0,
                pQueueCreateInfos=queueCreateInfos,
                queueCreateInfoCount=len(queueCreateInfos),
                pEnabledFeatures=[deviceFeatures],
                enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=len(validationLayers),
                ppEnabledLayerNames=validationLayers
            )
        else:
            createInfo = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                flags=0,
                pQueueCreateInfos=queueCreateInfos,
                queueCreateInfoCount=len(queueCreateInfos),
                pEnabledFeatures=[deviceFeatures],
                enabledExtensionCount=len(deviceExtensions),
                ppEnabledExtensionNames=deviceExtensions,
                enabledLayerCount=0
            )

        self.__device = vkCreateDevice(self.__physicalDevice, createInfo, None)
        if self.__device is None:
            raise Exception("failed to create logical device!")
        self.__graphicsQueue = vkGetDeviceQueue(self.__device, indices.graphicsFamily, 0)
        self.__presentQueue = vkGetDeviceQueue(self.__device, indices.presentFamily, 0)

    def __createSwapChain(self):
        swapChainSupport = self.__querySwapChainSupport(self.__physicalDevice)

        surfaceFormat = self.__chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = self.__chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = self.__chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount:
            imageCount = swapChainSupport.capabilities.maxImageCount

        createInfo = VkSwapchainCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            flags=0,
            surface=self.__surface,
            minImageCount=imageCount,
            imageFormat=surfaceFormat.format,
            imageColorSpace=surfaceFormat.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        )

        indices = self.__findQueueFamilies(self.__physicalDevice)
        if indices.graphicsFamily != indices.presentFamily:
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT
            createInfo.queueFamilyIndexCount = 2
            createInfo.pQueueFamilyIndices = [indices.graphicsFamily, indices.presentFamily]
        else:
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        createInfo.presentMode = presentMode
        createInfo.clipped = True

        vkCreateSwapchainKHR = vkGetDeviceProcAddr(self.__device, 'vkCreateSwapchainKHR')
        self.__swapChain = vkCreateSwapchainKHR(self.__device, createInfo, None)

        vkGetSwapchainImagesKHR = vkGetDeviceProcAddr(self.__device, 'vkGetSwapchainImagesKHR')
        self.__swapChainImages = vkGetSwapchainImagesKHR(self.__device, self.__swapChain)

        self.__swapChainImageFormat = surfaceFormat.format
        self.__swapChainExtent = extent

    def __createImageViews(self):
        self.__swapChainImageViews = []
        components = VkComponentMapping(VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY)
        subresourceRange = VkImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT,
                                                   0, 1, 0, 1)
        for i, image in enumerate(self.__swapChainImages):
            createInfo = VkImageViewCreateInfo(
                sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                flags=0,
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.__swapChainImageFormat,
                components=components,
                subresourceRange=subresourceRange
            )
            self.__swapChainImageViews.append(vkCreateImageView(self.__device, createInfo, None))

    def __createRenderPass(self):
        colorAttachment = VkAttachmentDescription(
            format=self.__swapChainImageFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        )

        depthAttachment = VkAttachmentDescription(
            format=self.depthFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        colorAttachmentRef = VkAttachmentReference(
            attachment=0,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        depthAttachmentRef = VkAttachmentReference(
            attachment=1,
            layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        subPass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=colorAttachmentRef,
            pDepthStencilAttachment=[depthAttachmentRef]
        )

        dependency = VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        )

        renderPassInfo = VkRenderPassCreateInfo(
            pAttachments=[colorAttachment, depthAttachment],
            pSubpasses=[subPass],
            pDependencies=[dependency]
        )

        self.__renderPass = vkCreateRenderPass(self.__device, renderPassInfo, None)

    def __createDescriptorSetLayout(self):
        uboLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT
        )

        samplerLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=1,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT
        )

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pBindings=[uboLayoutBinding, samplerLayoutBinding]
        )

        self.__descriptorSetLayout = vkCreateDescriptorSetLayout(self.__device, layoutInfo, None)

    def __createGraphicsPipeline(self):
        path = os.path.dirname(os.path.abspath(__file__))
        vertShaderModule = self.__createShaderModule(os.path.join(path, 'shaders\\vert.spv'))
        fragShaderModule = self.__createShaderModule(os.path.join(path, 'shaders\\frag.spv'))

        vertShaderStageInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            # flags=0,
            stage=VK_SHADER_STAGE_VERTEX_BIT,
            module=vertShaderModule,
            pName='main'
        )

        fragShaderStageInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            # flags=0,
            stage=VK_SHADER_STAGE_FRAGMENT_BIT,
            module=fragShaderModule,
            pName='main'
        )

        shaderStages = [vertShaderStageInfo, fragShaderStageInfo]

        bindingDescription = Vertex.getBindingDescription()
        attributeDescription = Vertex.getAttributeDescriptions()

        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            # vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[bindingDescription],
            # vertexAttributeDescriptionCount=len(attributeDescription),
            pVertexAttributeDescriptions=attributeDescription,
        )

        inputAssembly = VkPipelineInputAssemblyStateCreateInfo(
            # sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=False
        )

        viewport = VkViewport(0.0, 0.0,
                              float(self.__swapChainExtent.width),
                              float(self.__swapChainExtent.height),
                              0.0, 1.0)
        scissor = VkRect2D([0, 0], self.__swapChainExtent)
        viewportState = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=viewport,
            scissorCount=1,
            pScissors=scissor
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=VK_CULL_MODE_BACK_BIT,
            frontFace=VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=False
        )

        multisampling = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=False,
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT
        )

        depthStencil = VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=True,
            depthWriteEnable=True,
            depthCompareOp=VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=False,
            stencilTestEnable=False
        )

        colorBlendAttachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=False
        )

        colorBlending = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=colorBlendAttachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )

        pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            # setLayoutCount=0,
            pushConstantRangeCount=0,
            pSetLayouts=[self.__descriptorSetLayout]
        )

        self.__pipelineLayout = vkCreatePipelineLayout(self.__device, pipelineLayoutInfo, None)

        pipelineInfo = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            # stageCount=len(shaderStageInfos),
            pStages=shaderStages,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportState,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=colorBlending,
            pDepthStencilState=depthStencil,
            layout=self.__pipelineLayout,
            renderPass=self.__renderPass,
            subpass=0,
            basePipelineHandle=VK_NULL_HANDLE
        )

        self.__graphicsPipeline = vkCreateGraphicsPipelines(self.__device, VK_NULL_HANDLE, 1, pipelineInfo, None)[0]

        vkDestroyShaderModule(self.__device, vertShaderModule, None)
        vkDestroyShaderModule(self.__device, fragShaderModule, None)

    def __createFramebuffers(self):
        self.__swapChainFramebuffers = []

        for i, iv in enumerate(self.__swapChainImageViews):
            framebufferInfo = VkFramebufferCreateInfo(
                renderPass=self.__renderPass,
                pAttachments=[iv, self.__depthImageView],
                width=self.__swapChainExtent.width,
                height=self.__swapChainExtent.height,
                layers=1
            )

            self.__swapChainFramebuffers.append(vkCreateFramebuffer(self.__device, framebufferInfo, None))

    def __createCommandPool(self):
        queueFamilyIndices = self.__findQueueFamilies(self.__physicalDevice)

        poolInfo = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queueFamilyIndices.graphicsFamily
        )

        self.__commandPool = vkCreateCommandPool(self.__device, poolInfo, None)

    def __createDepthResources(self):
        depthFormat = self.depthFormat

        self.__depthImage, self.__depthImageMemory = self.__createImage(self.__swapChainExtent.width,
                                                                        self.__swapChainExtent.height,
                                                                        depthFormat,
                                                                        VK_IMAGE_TILING_OPTIMAL,
                                                                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.__depthImageView = self.__createImageView(self.__depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT)

        self.__transitionImageLayout(self.__depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)

    def __findSupportedFormat(self, candidates, tiling, feature):
        for i in candidates:
            props = vkGetPhysicalDeviceFormatProperties(self.__physicalDevice, i)

            if tiling == VK_IMAGE_TILING_LINEAR and (props.linearTilingFeatures & feature == feature):
                return i
            elif tiling == VK_IMAGE_TILING_OPTIMAL and (props.optimalTilingFeatures & feature == feature):
                return i
        return -1

    @property
    def depthFormat(self):
        return self.__findSupportedFormat(
            [VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT],
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)

    def hasStencilComponent(self, fm):
        return fm == VK_FORMAT_D32_SFLOAT_S8_UINT or fm == VK_FORMAT_D24_UNORM_S8_UINT

    def __createTextureImage(self):
        _image = Image.open(self.texturePath)
        _image.putalpha(1)
        width = _image.width
        height = _image.height
        imageSize = width * height * 4

        stagingBuffer, stagingMem = self.__createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingMem, 0, imageSize, 0)
        ffi.memmove(data, _image.tobytes(), imageSize)
        vkUnmapMemory(self.__device, stagingMem)

        del _image

        self.__textureImage, self.__textureImageMemory = self.__createImage(width, height,
                                                                            VK_FORMAT_R8G8B8A8_UNORM,
                                                                            VK_IMAGE_TILING_OPTIMAL,
                                                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__transitionImageLayout(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        self.__copyBufferToImage(stagingBuffer, self.__textureImage, width, height)
        self.__transitionImageLayout(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        vkDestroyBuffer(self.__device, stagingBuffer, None)
        vkFreeMemory(self.__device, stagingMem, None)

    def __createTextureImageView(self):
        self.__textureImageView = self.__createImageView(self.__textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                                         VK_IMAGE_ASPECT_COLOR_BIT)

    def __createTextureSampler(self):
        samplerInfo = VkSamplerCreateInfo(
            magFilter=VK_FILTER_LINEAR,
            minFilter=VK_FILTER_LINEAR,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            anisotropyEnable=True,
            maxAnisotropy=16,
            compareEnable=False,
            compareOp=VK_COMPARE_OP_ALWAYS,
            borderColor=VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalizedCoordinates=False
        )

        self.__textureSampler = vkCreateSampler(self.__device, samplerInfo, None)

    def __createImageView(self, image, imFormat, aspectFlage):
        ssr = VkImageSubresourceRange(
            aspectFlage,
            0, 1, 0, 1
        )

        viewInfo = VkImageViewCreateInfo(
            image=image,
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=imFormat,
            subresourceRange=ssr
        )

        return vkCreateImageView(self.__device, viewInfo, None)

    def __createImage(self, widht, height, imFormat, tiling, usage, properties):
        imageInfo = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D,
            extent=[widht, height, 1],
            mipLevels=1,
            arrayLayers=1,
            format=imFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=tiling,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED
        )

        image = vkCreateImage(self.__device, imageInfo, None)

        memReuirements = vkGetImageMemoryRequirements(self.__device, image)
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memReuirements.size,
            memoryTypeIndex=self.__findMemoryType(memReuirements.memoryTypeBits, properties)
        )

        imageMemory = vkAllocateMemory(self.__device, allocInfo, None)

        vkBindImageMemory(self.__device, image, imageMemory, 0)

        return (image, imageMemory)

    def __transitionImageLayout(self, image, imFormat, oldLayout, newLayout):
        cmdBuffer = self.__beginSingleTimeCommands()

        subresourceRange = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        if newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT
            if self.hasStencilComponent(imFormat):
                subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT

        barrier = VkImageMemoryBarrier(
            oldLayout=oldLayout,
            newLayout=newLayout,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=subresourceRange
        )

        sourceStage = 0
        destinationStage = 0

        if oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT
        elif oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        elif oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
        else:
            raise Exception('unsupported layout transition!')

        vkCmdPipelineBarrier(cmdBuffer,
                             sourceStage,
                             destinationStage,
                             0,
                             0, None,
                             0, None,
                             1, barrier)

        self.__endSingleTimeCommands(cmdBuffer)

    def __copyBufferToImage(self, buffer, image, width, height):
        cmdbuffer = self.__beginSingleTimeCommands()

        subresource = VkImageSubresourceLayers(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            mipLevel=0,
            baseArrayLayer=0,
            layerCount=1
        )
        region = VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=subresource,
            imageOffset=[0, 0],
            imageExtent=[width, height, 1]
        )

        vkCmdCopyBufferToImage(cmdbuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, region)

        self.__endSingleTimeCommands(cmdbuffer)

    def __loadModel(self):
        # startTime = time.time()
        # model = tol.LoadObj(self.modelPath)

        reader = tol.ObjReader()
        # Load .obj(and .mtl) using default configuration
        ret = reader.ParseFromFile(self.modelPath)
        attrib = reader.GetAttrib()
        vertices = attrib.vertices
        texcoords = attrib.texcoords
        materials = reader.GetMaterials()
        shapes = reader.GetShapes()

        uniqueVertices = {}
        vertexData = []
        indexData = []
        for shape in shapes:
            # allIndices = format(len(shape.mesh.indices))
            for i, idx in enumerate(shape.mesh.indices):
                # vid = allIndices[idx]
                vid = idx.vertex_index
                # tid = allIndices[idx+2]
                tid = idx.texcoord_index
                data = (
                    # vertex position
                    vertices[3 * vid + 0],
                    vertices[3 * vid + 1],
                    vertices[3 * vid + 2],

                    # color
                    1.0, 1.0, 1.0,

                    # texture coord
                    texcoords[2 * tid + 0],
                    1.0 - texcoords[2 * tid + 1]
                )

                if data not in uniqueVertices:
                    uniqueVertices[data] = len(vertexData)
                    vertexData.append(data)
                indexData.append(uniqueVertices[data])

        # useTime = time.time() - startTime
        # print('Model loading time: {} s'.format(useTime))
        self.__vertices = np.array(vertexData, np.float32)
        self.__indices = np.array(indexData, np.uint32)

    def __createVertexBuffer(self):
        bufferSize = self.__vertices.nbytes

        stagingBuffer, stagingMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        data = vkMapMemory(self.__device, stagingMemory, 0, bufferSize, 0)
        vertePtr = ffi.cast('float *', self.__vertices.ctypes.data)
        ffi.memmove(data, vertePtr, bufferSize)
        vkUnmapMemory(self.__device, stagingMemory)

        self.__vertexBuffer, self.__vertexBufferMemory = self.__createBuffer(bufferSize,
                                                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.__copyBuffer(stagingBuffer, self.__vertexBuffer, bufferSize)

        vkDestroyBuffer(self.__device, stagingBuffer, None)
        vkFreeMemory(self.__device, stagingMemory, None)

    def __createBuffer(self, size, usage, properties):
        buffer = None
        bufferMemory = None

        bufferInfo = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = vkCreateBuffer(self.__device, bufferInfo, None)

        memRequirements = vkGetBufferMemoryRequirements(self.__device, buffer)
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(memRequirements.memoryTypeBits, properties)
        )
        bufferMemory = vkAllocateMemory(self.__device, allocInfo, None)

        vkBindBufferMemory(self.__device, buffer, bufferMemory, 0)

        return (buffer, bufferMemory)

    def __beginSingleTimeCommands(self):
        allocInfo = VkCommandBufferAllocateInfo(
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.__commandPool,
            commandBufferCount=1
        )

        commandBuffer = vkAllocateCommandBuffers(self.__device, allocInfo)[0]
        beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vkBeginCommandBuffer(commandBuffer, beginInfo)

        return commandBuffer

    def __endSingleTimeCommands(self, commandBuffer):
        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(pCommandBuffers=[commandBuffer])

        vkQueueSubmit(self.__graphicsQueue, 1, [submitInfo], VK_NULL_HANDLE)
        vkQueueWaitIdle(self.__graphicsQueue)

        vkFreeCommandBuffers(self.__device, self.__commandPool, 1, [commandBuffer])

    def __copyBuffer(self, src, dst, bufferSize):
        allocInfo = VkCommandBufferAllocateInfo(
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.__commandPool,
            commandBufferCount=1
        )

        commandBuffer = vkAllocateCommandBuffers(self.__device, allocInfo)[0]
        beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vkBeginCommandBuffer(commandBuffer, beginInfo)

        # copyRegion = VkBufferCopy(size=bufferSize)
        copyRegion = VkBufferCopy(0, 0, bufferSize)
        vkCmdCopyBuffer(commandBuffer, src, dst, 1, [copyRegion])

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(pCommandBuffers=[commandBuffer])

        vkQueueSubmit(self.__graphicsQueue, 1, [submitInfo], VK_NULL_HANDLE)
        vkQueueWaitIdle(self.__graphicsQueue)

        vkFreeCommandBuffers(self.__device, self.__commandPool, 1, [commandBuffer])

    def __findMemoryType(self, typeFilter, properties):
        memProperties = vkGetPhysicalDeviceMemoryProperties(self.__physicalDevice)

        for i, prop in enumerate(memProperties.memoryTypes):
            if (typeFilter & (1 << i)) and ((prop.propertyFlags & properties) == properties):
                return i

        return -1

    def __createIndexBuffer(self):
        bufferSize = self.__indices.nbytes

        stagingBuffer, stagingMemory = self.__createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        data = vkMapMemory(self.__device, stagingMemory, 0, bufferSize, 0)
        indicesPtr = ffi.cast('uint16_t*', self.__indices.ctypes.data)
        ffi.memmove(data, indicesPtr, bufferSize)
        vkUnmapMemory(self.__device, stagingMemory)

        self.__indexBuffer, self.__indexBufferMemory = self.__createBuffer(bufferSize,
                                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        self.__copyBuffer(stagingBuffer, self.__indexBuffer, bufferSize)

        vkDestroyBuffer(self.__device, stagingBuffer, None)
        vkFreeMemory(self.__device, stagingMemory, None)

    def __createUniformBuffer(self):
        self.__uniformBuffer, self.__uniformBufferMemory = self.__createBuffer(self.__ubo.nbytes,
                                                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

    def __createDescriptorPool(self):
        poolSize1 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1
        )

        poolSize2 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1
        )

        poolInfo = VkDescriptorPoolCreateInfo(
            pPoolSizes=[poolSize1, poolSize2],
            maxSets=1
        )

        self.__descriptorPool = vkCreateDescriptorPool(self.__device, poolInfo, None)

    def __createDescriptorSet(self):
        layouts = [self.__descriptorSetLayout]
        allocInfo = VkDescriptorSetAllocateInfo(
            descriptorPool=self.__descriptorPool,
            pSetLayouts=layouts
        )

        self.__descriptorSet = vkAllocateDescriptorSets(self.__device, allocInfo)

        bufferInfo = VkDescriptorBufferInfo(
            buffer=self.__uniformBuffer,
            offset=0,
            range=self.__ubo.nbytes
        )

        imageInfo = VkDescriptorImageInfo(
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            imageView=self.__textureImageView,
            sampler=self.__textureSampler
        )

        descriptWrite1 = VkWriteDescriptorSet(
            dstSet=self.__descriptorSet[0],
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            # descriptorCount=1,
            pBufferInfo=[bufferInfo]
        )

        descriptWrite2 = VkWriteDescriptorSet(
            dstSet=self.__descriptorSet[0],
            dstBinding=1,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo=[imageInfo]
        )

        vkUpdateDescriptorSets(self.__device, 2, [descriptWrite1, descriptWrite2], 0, None)

    def __createCommandBuffers(self):
        # self.__commandBuffers = []

        allocInfo = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__swapChainFramebuffers)
        )

        self.__commandBuffers = vkAllocateCommandBuffers(self.__device, allocInfo)
        # self.__commandBuffers = [ffi.addressof(commandBuffers, i)[0] for i in range(len(self.__swapChainFramebuffers))]

        for i, cmdBuffer in enumerate(self.__commandBuffers):
            beginInfo = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
            )

            vkBeginCommandBuffer(cmdBuffer, beginInfo)

            # clearColor = VkClearValue([[0.0, 0.0, 0.0, 1.0]])
            renderArea = VkRect2D([0, 0], self.__swapChainExtent)
            clearColor = [VkClearValue(color=[[0.0, 0.0, 0.0, 1.0]]), VkClearValue(depthStencil=[1.0, 0])]
            renderPassInfo = VkRenderPassBeginInfo(
                sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.__renderPass,
                framebuffer=self.__swapChainFramebuffers[i],
                renderArea=renderArea,
                pClearValues=clearColor
            )

            # renderPassInfo.clearValueCount = 1
            # renderPassInfo.pClearValues = ffi.addressof(clearColor)

            vkCmdBeginRenderPass(cmdBuffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__graphicsPipeline)

            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, [self.__vertexBuffer], [0])

            # vkCmdBindIndexBuffer(cmdBuffer, self.__indexBuffer, 0, VK_INDEX_TYPE_UINT16)
            vkCmdBindIndexBuffer(cmdBuffer, self.__indexBuffer, 0, VK_INDEX_TYPE_UINT32)

            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__pipelineLayout, 0, 1,
                                    self.__descriptorSet, 0, None)

            vkCmdDrawIndexed(cmdBuffer, len(self.__indices), 1, 0, 0, 0)

            vkCmdEndRenderPass(cmdBuffer)

            vkEndCommandBuffer(cmdBuffer)

    def __createSemaphores(self):
        semaphoreInfo = VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)

        self.__imageAvailableSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)
        self.__renderFinishedSemaphore = vkCreateSemaphore(self.__device, semaphoreInfo, None)

    def __updateProjectionMatrix(self):
        aspectRatio = float(self.__swapChainExtent.width) / float(self.__swapChainExtent.height)
        half_tan_fov = (np.tan(45 / 2))
        far = 10
        near = 0.1

        self.__ubo.proj[0][0] = aspectRatio / half_tan_fov
        self.__ubo.proj[1][1] = 1.0 / half_tan_fov
        self.__ubo.proj[2][2] = (far) / (far - near)
        self.__ubo.proj[2][3] = -(near * far) / (far - near)
        self.__ubo.proj[3][2] = 1
        self.__ubo.proj[3][3] = 2

    def __updateUniformBuffer(self):
        currentTime = time.time()

        t = currentTime - self.__startTime

        # Set model matrix to an identity matrix (no rotation)
        self.__ubo.model = np.identity(4, np.float32)

        # View matrix (camera setup remains the same)
        self.__ubo.view = glm.lookAt(
            np.array([2, 2, 2], np.float32),
            np.array([0, 0, 0], np.float32),
            np.array([0, 0, -1], np.float32)
        )

        # Update projection matrix (if necessary)
        self.__updateProjectionMatrix()

        # Convert matrices to NumPy arrays for GPU upload
        self.__ubo.model = np.array(self.__ubo.model)
        self.__ubo.view = np.array(self.__ubo.view)

        # Map the uniform buffer memory and upload data
        data = vkMapMemory(self.__device, self.__uniformBufferMemory, 0, self.__ubo.nbytes, 0)
        ma = self.__ubo.toArray()
        dptr = ffi.cast('float *', ma.ctypes.data)
        ffi.memmove(data, dptr, self.__ubo.nbytes)

        # Unmap the memory
        vkUnmapMemory(self.__device, self.__uniformBufferMemory)

    def __drawFrame(self):
        self.__updateUniformBuffer()
        vkAcquireNextImageKHR = vkGetDeviceProcAddr(self.__device, 'vkAcquireNextImageKHR')
        vkQueuePresentKHR = vkGetDeviceProcAddr(self.__device, 'vkQueuePresentKHR')

        try:
            imageIndex = vkAcquireNextImageKHR(self.__device, self.__swapChain, 18446744073709551615,
                                               self.__imageAvailableSemaphore, VK_NULL_HANDLE)
        except VkErrorSurfaceLostKhr:
            print('Surface lost')
            return

        submitInfo = VkSubmitInfo(sType=VK_STRUCTURE_TYPE_SUBMIT_INFO)

        waitSemaphores = ffi.new('VkSemaphore[]', [self.__imageAvailableSemaphore])
        waitStages = ffi.new('uint32_t[]', [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, ])
        submitInfo.waitSemaphoreCount = 1
        submitInfo.pWaitSemaphores = waitSemaphores
        submitInfo.pWaitDstStageMask = waitStages

        cmdBuffers = ffi.new('VkCommandBuffer[]', [self.__commandBuffers[imageIndex], ])
        submitInfo.commandBufferCount = 1
        submitInfo.pCommandBuffers = cmdBuffers

        signalSemaphores = ffi.new('VkSemaphore[]', [self.__renderFinishedSemaphore])
        submitInfo.signalSemaphoreCount = 1
        submitInfo.pSignalSemaphores = signalSemaphores

        vkQueueSubmit(self.__graphicsQueue, 1, submitInfo, VK_NULL_HANDLE)

        swapChains = [self.__swapChain]
        presentInfo = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            # waitSemaphoreCount=1,
            pWaitSemaphores=signalSemaphores,
            # swapchainCount=1,
            pSwapchains=swapChains,
            pImageIndices=[imageIndex]
        )

        vkQueuePresentKHR(self.__presentQueue, presentInfo)

    def __createShaderModule(self, shaderFile):
        with open(shaderFile, 'rb') as sf:
            code = sf.read()
            codeSize = len(code)

            createInfo = VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=codeSize,
                pCode=code
            )

            return vkCreateShaderModule(self.__device, createInfo, None)

    def __chooseSwapSurfaceFormat(self, availableFormats):
        if len(availableFormats) == 1 and availableFormats[0].format == VK_FORMAT_UNDEFINED:
            return VkSurfaceFormatKHR(VK_FORMAT_B8G8R8A8_UNORM, 0)

        for availableFormat in availableFormats:
            if availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM and availableFormat.colorSpace == 0:
                return availableFormat

        return availableFormats[0]

    def __chooseSwapPresentMode(self, availablePresentModes):
        for availablePresentMode in availablePresentModes:
            if availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR:
                return availablePresentMode

        return VK_PRESENT_MODE_FIFO_KHR

    def __chooseSwapExtent(self, capabilities):
        width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, WIDTH))
        height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, HEIGHT))
        return VkExtent2D(width, height)

    def __querySwapChainSupport(self, device):
        details = SwapChainSupportDetails()

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vkGetInstanceProcAddr(self.__instance,
                                                                          'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        details.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, self.__surface)

        vkGetPhysicalDeviceSurfaceFormatsKHR = vkGetInstanceProcAddr(self.__instance,
                                                                     'vkGetPhysicalDeviceSurfaceFormatsKHR')
        details.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, self.__surface)

        vkGetPhysicalDeviceSurfacePresentModesKHR = vkGetInstanceProcAddr(self.__instance,
                                                                          'vkGetPhysicalDeviceSurfacePresentModesKHR')
        details.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(device, self.__surface)

        return details

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)
        extensionsSupported = self.__checkDeviceExtensionSupport(device)
        swapChainAdequate = False
        if extensionsSupported:
            swapChainSupport = self.__querySwapChainSupport(device)
            swapChainAdequate = (not swapChainSupport.formats is None) and (not swapChainSupport.presentModes is None)

        supportedFeatures = vkGetPhysicalDeviceFeatures(device)
        return indices.isComplete() and extensionsSupported and swapChainAdequate and supportedFeatures.samplerAnisotropy

    def __checkDeviceExtensionSupport(self, device):
        availableExtensions = vkEnumerateDeviceExtensionProperties(device, None)

        for extension in availableExtensions:
            if extension.extensionName in deviceExtensions:
                return True

        return False

    def __findQueueFamilies(self, device):
        vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(self.__instance,
                                                                     'vkGetPhysicalDeviceSurfaceSupportKHR')
        indices = QueueFamilyIndices()

        queueFamilies = vkGetPhysicalDeviceQueueFamilyProperties(device)

        for i, queueFamily in enumerate(queueFamilies):
            if queueFamily.queueCount > 0 and queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, self.__surface)

            if queueFamily.queueCount > 0 and presentSupport:
                indices.presentFamily = i

            if indices.isComplete():
                break

        return indices

    def __getRequiredExtensions(self):
        extensions = list(map(str, glfw.get_required_instance_extensions()))

        if enableValidationLayers:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        return extensions

    def __checkValidationLayerSupport(self):
        availableLayers = vkEnumerateInstanceLayerProperties()
        for layerName in validationLayers:
            layerFound = False

            for layerProperties in availableLayers:
                if layerName == layerProperties.layerName:
                    layerFound = True
                    break
            if not layerFound:
                return False

        return True

    def run(self):
        self.__initWindow()
        self.__initVulkan()
        self.__mainLoop()


if __name__ == '__main__':
    app = HelloTriangleApplication()

    app.run()

    del app
    glfw.terminate()