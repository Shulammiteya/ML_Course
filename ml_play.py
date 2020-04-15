"""
The template of the main script of the machine learning process
"""
import pickle
from os import path

import numpy as np
import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)

def ml_loop():
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False
    filename = path.join(path.dirname(__file__), 'save', 'clf_KMeans_BallAndDirection.pickle')
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    s = [93, 93]

    def get_direction(ball_x, ball_y, ball_pre_x, ball_pre_y):
        VectorX = ball_x - ball_pre_x
        VectorY = ball_y - ball_pre_y
        if (VectorX >= 0 and VectorY >= 0):
            return 0
        elif (VectorX > 0 and VectorY < 0):
            return 1
        elif (VectorX < 0 and VectorY > 0):
            return 2
        elif (VectorX < 0 and VectorY < 0):
            return 3

    def get_point(ball_x, ball_y, dir, dir_x_abs):
        if (dir_x_abs == 0):
            return scene_info.platform[0]
        crush = -1
        while (ball_y < 400):
            ball_x = ball_x + dir
            ball_y = ball_y + dir_x_abs
            if (ball_x < 0):
                ball_x = 0
                dir = -dir
                crush = crush * -1
            elif (ball_x > 200):
                ball_x =200
                dir = -dir
                crush = crush * -1
        while (ball_y > 400):
            ball_x = ball_x - dir/dir_x_abs
            ball_y = ball_y - 1
        return ball_x * crush

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        feature = []
        feature.append(scene_info.ball[0])
        feature.append(scene_info.ball[1])
        feature.append(scene_info.platform[0])
        
        dir = feature[0] - s[0]
        dir_x_abs = abs(dir)
        direction = get_direction(feature[0],feature[1],s[0],s[1])
        feature.append(direction)
        s = [feature[0], feature[1]]
        feature = np.array(feature)
        feature = feature.reshape((-1,4))

        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            ball_served = False

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_LEFT)
            ball_served = True
        else:
                
            """y = clf.predict(feature)
            
            if y == 0:
                instruction = 'NONE'
            elif y == 1:
                instruction = 'LEFT'
            elif y == 2:
                instruction = 'RIGHT'"""
            
            instruction = 'NONE'
            if (scene_info.ball[1] > 255):#and (direction == 0 or direction == 2)):
                point =  get_point(scene_info.ball[0], scene_info.ball[1], dir, dir_x_abs)
                if (point < 0):
                    point = -point
                elif (direction == 0):
                    direction = 2
                else:
                    direction = 0
                if (direction == 0):
                    if (point - scene_info.platform[0] > 39):
                        instruction = 'RIGHT'
                    elif (point - scene_info.platform[0] < 28):
                        instruction = 'LEFT'
                else:
                    if (point - scene_info.platform[0] < 1):
                        instruction = 'LEFT'
                    elif (point - scene_info.platform[0] > 12):
                        instruction = 'RIGHT'
            elif (scene_info.ball[1] > 200 ):#and (direction == 0 or direction == 2)):
                if (scene_info.platform[0] < 60):
                    instruction = 'RIGHT'
                elif (scene_info.platform[0] > 90):
                    instruction = 'LEFT'

            if instruction == 'LEFT':
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                #print('LEFT')
            elif instruction == 'RIGHT':
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                #print('RIGHT')
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)
                #print('NONE')

