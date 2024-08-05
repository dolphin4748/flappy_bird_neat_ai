import neat.config
import neat.nn.feed_forward
import neat.population
import pygame as pg
import neat
import random
import os
import time

pg.font.init()

width, height = 576, 750

bird_img = [pg.transform.scale2x(pg.image.load(os.path.join("imgs", "bird1.png"))),
             pg.transform.scale2x(pg.image.load(os.path.join("imgs", "bird2.png"))),
             pg.transform.scale2x(pg.image.load(os.path.join("imgs", "bird3.png")))
            ]

pipe_img = pg.transform.scale2x(pg.image.load(os.path.join("imgs", "pipe.png")))
base_img = pg.transform.scale2x(pg.image.load(os.path.join("imgs", "base.png")))
back_img = pg.transform.scale2x(pg.image.load(os.path.join("imgs", "bg.png")))

stat_font = pg.font.SysFont('calibri', 50)

class Bird:
    IMGS = bird_img
    max_rotation = 25
    rotation_vel = 20
    animation_time = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -9.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count**2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
        else:
            if self.tilt > -90:
                self.tilt -= self.rotation_vel
    def draw(self, screen):
        self.img_count += 1

        if self.img_count < self.animation_time:
            self.img = self.IMGS[0]
        elif self.img_count < self.animation_time*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.animation_time*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.animation_time*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.animation_time*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.animation_time*2

        rotated_img = pg.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        screen.blit(rotated_img, new_rect.topleft)

    def get_mask(self):
        return pg.mask.from_surface(self.img)
    
class Pipe:
    gap = 250
    spd = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.pipe_top = pg.transform.flip(pipe_img, False, True)
        self.pipe_bottom = pipe_img

        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50, 400)
        self.top = self.height - self.pipe_top.get_height()
        self.bottom = self.height + self.gap
    
    def move(self):
        self.x -= self.spd


    def draw(self, screen):
        screen.blit(self.pipe_top, (self.x, self.top))
        screen.blit(self.pipe_bottom, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pg.mask.from_surface(self.pipe_top)
        bottom_mask = pg.mask.from_surface(self.pipe_bottom)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    spd = 5
    width = base_img.get_width()
    img = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width
    
    def move(self):
        self.x1 -= self.spd
        self.x2 -= self.spd

        if self.x1 <= 0 - self.width:
            self.x1 = width
        if self.x2 <= 0 - self.width:
            self.x2 = width
    
    def draw(self, screen):
        screen.blit(self.img, (self.x1, self.y))
        screen.blit(self.img, (self.x2, self.y))


    
def draw_window(screen, birds, pipes, base, score):
    screen.blit(back_img, (0, -100))
    for bird in birds:
        bird.draw(screen)

    for pipe in pipes:
        pipe.draw(screen)
    
    base.draw(screen)

    population = len(birds)
    
    text = stat_font.render("score: " + str(score), True, (255, 255, 255))
    text_p = stat_font.render("population: " + str(population), True, (255, 255, 255))
    screen.blit(text, (width - 10 - text.get_width(), 10))
    screen.blit(text_p, (10, 10))

    pg.display.update()

def main(genomes, config):
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(100, 200))
        g.fitness = 0
        ge.append(g)
    
    pipes = [Pipe(576)]
    base = Base(650)
    add_pipe = False

    score = 0

    screen = pg.display.set_mode((width, height))
    clock = pg.time.Clock()
    running = True

    while running:
        clock.tick(60)

        key = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.QUIT or key[pg.K_ESCAPE]:
                running = False
                pg.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                pipe_ind = 1
        else:
            score = 0
            running = False
            break

        remove_list = []
        
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.05

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

            if bird.y + bird.img.get_height()/2 > 650 or bird.y < 0:
                ge.pop(x)
                birds.pop(x)
                nets.pop(x)
                
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    ge.pop(x)
                    birds.pop(x)
                    nets.pop(x)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            pipe.move()

            if pipe.x <= 0 - pipe.pipe_top.get_width():
                remove_list.append(pipe)
        if add_pipe == True:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(576))
            add_pipe = False

        for i in remove_list:
            pipes.remove(i)

        base.move()
        draw_window(screen, birds, pipes, base, score)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)