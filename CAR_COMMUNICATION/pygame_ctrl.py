import pygame
#from pygame.local import *

HEIGHT = 500
WIDTH = 500

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    clock = pygame.time.Clock()

    pygame.key.set_repeat(1,1)

    while True:
        try:
            pressed = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if pressed[pygame.K_UP]:
                         print("pressed UP      ", sep='', end='/r', flush=True)
                    if pressed[pygame.K_DOWN]:
                         print("pressed DOWN    ", sep='', end='/r', flush=True)
                    if pressed[pygame.K_RIGHT]:
                         print("pressed RIGHT   ", sep='', end='/r', flush=True)
                    if pressed[pygame.K_LEFT]:
                         print("pressed LEFT    ", sep='', end='/r', flush=True)
           
            screen.fill((0,0,0))
            pygame.display.flip()
            clock.tick(60)

        except KeyboardInterrupt:
            break

main()

        
