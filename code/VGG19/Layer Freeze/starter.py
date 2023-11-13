from discordwebhook import Discord
import time
import pycuda.driver as cuda

# Initialize CUDA
cuda.init()
device = cuda.Device(0)  # Use 0 for the first GPU, 1 for the second, and so on


def DiscordNotification(Msg):
    webHookUrl = "https://discord.com/api/webhooks/1132597585824202813/8XDNjpwwOIsistL4nThyY7NjVo67UVHckbtOAAdGAf96_TZ7dTS3tOpDmle646rF_ZDX"
    discord = Discord(url=webHookUrl)
    discord.post(content=Msg)


try:
    # Create a CUDA context
    context = device.make_context()

    def runPythonScript():
        # Replace this with the code to run your script
        DiscordNotification("Server free now run script")

    def getRate():
        # Get GPU memory information
        total_memory = device.total_memory()
        free_memory, _ = cuda.mem_get_info()

        # Calculate used memory and usage percentage
        used_memory = total_memory - free_memory
        memory_usage_percentage = (used_memory / total_memory) * 100

        print(f"GPU Memory Usage: {memory_usage_percentage:.2f}%")
        return memory_usage_percentage

    while True:
        rate = getRate()
        if rate < 93:
            runPythonScript()
            break
        time.sleep(60)
finally:
    # Clean up the CUDA context
    DiscordNotification("Server checker exist")
    context.pop()
