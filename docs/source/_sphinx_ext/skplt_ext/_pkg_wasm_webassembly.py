import micropip
import asyncio


async def install_packages():
    await micropip.install("scikit-plots==0.3.9rc3", keep_going=True)


asyncio.run(install_packages())  # Directly run the async function
