import pygame
import sys
from Assistant import AIRA
from text_to_speech import STT
pygame.init()

WIDTH, HEIGHT = 800, 400 # of GUI
FONT_SIZE = 25
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Desktop Assistant")
font_list = ['./Pixellettersfull-BnJ5.ttf', './AovelSansRounded-rdDL.ttf', './Mvdawlatulislam-Z8dJ.ttf', './ShortBaby-Mg2w.ttf']
font_file = font_list[0] # Choose the font of your choice
font = pygame.font.Font(font_file, FONT_SIZE)

messages = []

def draw_text(text, font, color, surface, x, y):
    text_object = font.render(text, True, color)
    text_rect = text_object.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_object, text_rect)

def main():
    clock = pygame.time.Clock()
    model = AIRA()

    input_text = ""
    input_active = True

    scroll_offset = 4
    model_turn = False
    obj = STT()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # if event.type == pygame.KEYDOWN:
            if not model_turn:
                # record audio                    
                try:
                    voice = obj.listen()
                    inp = obj.transcribe(voice)
                    print(inp)
                    messages.append(("USER: ", inp))
                    model_turn = True              
                    screen.fill(BLACK)
                    draw_messages(scroll_offset)    
                    pygame.display.flip() 
                    
                except KeyboardInterrupt:
                    break
            
            if model_turn:        
                gen = model.run(inp)
                next_word = ''
                total_text = ''
                messages.append('')
                first_word = True
                while True:
                    try:
                        if first_word:
                            old_len = len(next_word)
                            next_word = next(gen)
                            first_word = False
                            
                        old_len = len(next_word)
                        next_word = next(gen)
                        total_text += next_word[old_len:]
                        if '<|im_start|> assistant' in total_text:
                            index = total_text.index('<|im_start|> assistant')
                            total_text = total_text[index + 22:]
                        messages.pop()
                        messages.append(("Assistant: ", total_text))
                        screen.fill(BLACK)
                        draw_messages(scroll_offset)    
                        pygame.display.flip()
                    except StopIteration:
                        break
                model_turn = False
                        

                        
        screen.fill(BLACK)
        draw_messages(scroll_offset)

        if input_active:
            pygame.draw.rect(screen, WHITE, (10, HEIGHT - 40, WIDTH - 20, 30))
            draw_text(input_text, font, BLACK, screen, 20, HEIGHT - 35)

        pygame.display.flip()
        clock.tick(60)

def draw_messages(scroll_offset):
    y = HEIGHT - 50
    max_line_width = WIDTH - 40 
    for sender, message in reversed(messages):
        full_text = f"{sender}: {message}"
        words = full_text.split()
        lines = []
        current_line = ""

        for word in words:
            if font.size(current_line + word)[0] <= max_line_width:
                current_line += word + " "
                
            else:
                lines.append(current_line)
                current_line = word + " "
        
        lines.append(current_line)
        for line in reversed(lines):
            text_surface = font.render(line, True, WHITE)
            text_rect = text_surface.get_rect()
            text_rect.bottomleft = (20, y - scroll_offset)
            screen.blit(text_surface, text_rect)
            y -= text_rect.height + 5

if __name__ == "__main__":
    main()
