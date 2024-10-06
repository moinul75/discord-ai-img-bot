import os 
import discord 
import torch 
import uuid 
import asyncio
from PIL import Image
from diffusers import StableDiffusionPipeline  
from dotenv import load_dotenv

load_dotenv()

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

token = os.environ.get('TOKEN') 
print("token: ", token)

intents = discord.Intents.default() 
intents.messages = True 
intents.message_content = True  

client = discord.Client(intents=intents) 

@client.event 
async def on_connect(): 
    print('Bot Connected')

@client.event 
async def on_message(message): 
    if message.author == client.user: 
        return
    
    if message.content.startswith('!gen '): 
        await message.channel.send(f'Generating image from prompt: "{message.content[5:]}"....') 
        
        filename = f'{uuid.uuid4()}.png'  
        prompt = message.content[5:] 
        height = 152 
        width = 152  
        num_inference_steps = 25 


        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, lambda: pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps).images[0])
        image = image.resize((250, 250))
        image.save(filename) 
        
        with open(filename, 'rb') as f:  
            pic = discord.File(f)
            await message.channel.send(file=pic)  
            
        os.remove(filename)  
            
client.run(token)
