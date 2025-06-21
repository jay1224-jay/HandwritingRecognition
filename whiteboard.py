import pygame
import pygame.locals
print("=== Loading and Initializing model... ===")
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy  as np


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Model is on ", device)
model = CNN().to(device)
model.load_state_dict(torch.load('my_model.pth', map_location='cuda', weights_only=True))
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1] and shape [1, 28, 28]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])



print("=== Model loaded and initialized successfully! ===")
pygame.init()

WIDTH = 800
HEIGHT = 600
surface = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = pygame.time.Clock()
FPS.tick(60)

CLICK       = 1
RELEASE     = 2
state       = 0
mouse_x     = 0
mouse_y     = 0
mouse_pos   = 0
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
GAME_FONT = pygame.font.SysFont('Consolas', 30)

blocks = [[ 0 for _ in range(28) ] for i in range(28) ]

border_height = 560
border_width  = 560
border = pygame.Rect(200 + (600-border_width)/2, (HEIGHT-border_height)/2, 
                     border_width, border_height)

def f():
    print("called f")


class Button:
    def __init__(self, x, y, width, height, text="button", command=f):
        self.x      = x
        self.y      = y
        self.width  = width
        self.height = height
        self.command=command
        self.rect_obj = pygame.Rect(x, y, self.width, self.height)
        self.text_surface_object = GAME_FONT.render(text, True, BLACK)
        self.text_rect = self.text_surface_object.get_rect(center=self.rect_obj.center)

    def draw(self):
        pygame.draw.rect(surface, WHITE, self.text_rect)
        surface.blit(self.text_surface_object, self.text_rect)

    def execute(self):
        self.command()

    def is_pressed(self, x, y):
        return (x >= self.x and x <= self.x + self.width and y >= self.y and y <= self.y+self.height)

predicted_digit = -1

def recognize_command():
    global predicted_digit
    img = np.array(blocks, dtype=np.float32)
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]
    # print(img_tensor)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_digit = int(torch.argmax(output, dim=1))
    # print("Prediction:", predicted_digit)

def clear_command():
    for y in range(28):
        for x in range(28):
            blocks[y][x] = 0

RecButton = Button(50, 50, 120, 50, "Recognize", recognize_command)
ClearButton = Button(50, HEIGHT-100, 100, 50, "Clear", clear_command)

while True:
    surface.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.locals.MOUSEBUTTONDOWN:
            state = CLICK
        elif event.type == pygame.locals.MOUSEBUTTONUP:
            state = RELEASE
    
    if state == CLICK:
        mouse_pos = pygame.mouse.get_pos()
        mouse_x = mouse_pos[0]
        mouse_y = mouse_pos[1]
        if ( mouse_x >= 220 and mouse_x <= 780 and mouse_y >= 20 and mouse_y <= 580 ):
            mouse_x -= 220
            mouse_y -= 20
            mouse_x = int(mouse_x/20)
            mouse_y = int(mouse_y/20)
            blocks[mouse_y][mouse_x] = 1
        elif ( RecButton.is_pressed(mouse_x, mouse_y) ):
            RecButton.execute()
        elif ( ClearButton.is_pressed(mouse_x, mouse_y) ):
            ClearButton.execute()


    pygame.draw.rect(surface, WHITE, border, 2)

    for y in range(28):
        for x in range(28):
            if ( blocks[y][x] ):
                pygame.draw.rect(surface, WHITE, pygame.Rect(220 + 20*x, 20 + 20*y, 20, 20))

    text_surface1 = GAME_FONT.render("Digit: %d" % predicted_digit, 1, WHITE)
    surface.blit(text_surface1, (30, 150))

    RecButton.draw()
    ClearButton.draw()

    pygame.display.update()