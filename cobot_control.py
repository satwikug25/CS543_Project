import re
from pymycobot import MyCobot280

# ———————————————
# 1) Initialize
# ———————————————
mc = MyCobot280('/dev/cu.usbserial-54F70028471')
mc.power_on()
mc.set_gripper_value(0, 50)  # ensure gripper open at start

# ———————————————
# 2) Define your home pose
#    (fill in with a safe “parked” 6-DOF pose)
# ———————————————
home_position = [0, 0, 0, 0, 0, 0]  # <-- X, Y, Z, RX, RY, RZ
def go_home(speed=50):
    mc.send_coords(home_position, speed, 0)

# ———————————————
# 3) Board lookup (0-based)
# ———————————————
board_positions = board_positions = {
    (0, 0): [140,  20,  30, 180, 0, 0],
    (0, 1): [140,  50,  30, 180, 0, 0],
    (0, 2): [140,  80,  30, 180, 0, 0],
    (0, 3): [140, 110,  30, 180, 0, 0],
    (0, 4): [140, 140,  30, 180, 0, 0],
    (0, 5): [140, 170,  30, 180, 0, 0],
    (0, 6): [140, 200,  30, 180, 0, 0],
    (0, 7): [140, 230,  30, 180, 0, 0],

    (1, 0): [150,  20,  30, 180, 0, 0],
    (1, 1): [150,  50,  30, 180, 0, 0],
    (1, 2): [150,  80,  30, 180, 0, 0],
    (1, 3): [150, 110,  30, 180, 0, 0],
    (1, 4): [150, 140,  30, 180, 0, 0],
    (1, 5): [150, 170,  30, 180, 0, 0],
    (1, 6): [150, 200,  30, 180, 0, 0],
    (1, 7): [150, 230,  30, 180, 0, 0],

    (2, 0): [160,  20,  30, 180, 0, 0],
    (2, 1): [160,  50,  30, 180, 0, 0],
    (2, 2): [160,  80,  30, 180, 0, 0],
    (2, 3): [160, 110,  30, 180, 0, 0],
    (2, 4): [160, 140,  30, 180, 0, 0],
    (2, 5): [160, 170,  30, 180, 0, 0],
    (2, 6): [160, 200,  30, 180, 0, 0],
    (2, 7): [160, 230,  30, 180, 0, 0],

    (3, 0): [170,  20,  30, 180, 0, 0],
    (3, 1): [170,  50,  30, 180, 0, 0],
    (3, 2): [170,  80,  30, 180, 0, 0],
    (3, 3): [170, 110,  30, 180, 0, 0],
    (3, 4): [170, 140,  30, 180, 0, 0],
    (3, 5): [170, 170,  30, 180, 0, 0],
    (3, 6): [170, 200,  30, 180, 0, 0],
    (3, 7): [170, 230,  30, 180, 0, 0],

    (4, 0): [180,  20,  30, 180, 0, 0],
    (4, 1): [180,  50,  30, 180, 0, 0],
    (4, 2): [180,  80,  30, 180, 0, 0],
    (4, 3): [180, 110,  30, 180, 0, 0],
    (4, 4): [180, 140,  30, 180, 0, 0],
    (4, 5): [180, 170,  30, 180, 0, 0],
    (4, 6): [180, 200,  30, 180, 0, 0],
    (4, 7): [180, 230,  30, 180, 0, 0],

    (5, 0): [190,  20,  30, 180, 0, 0],
    (5, 1): [190,  50,  30, 180, 0, 0],
    (5, 2): [190,  80,  30, 180, 0, 0],
    (5, 3): [190, 110,  30, 180, 0, 0],
    (5, 4): [190, 140,  30, 180, 0, 0],
    (5, 5): [190, 170,  30, 180, 0, 0],
    (5, 6): [190, 200,  30, 180, 0, 0],
    (5, 7): [190, 230,  30, 180, 0, 0],

    (6, 0): [200,  20,  30, 180, 0, 0],
    (6, 1): [200,  50,  30, 180, 0, 0],
    (6, 2): [200,  80,  30, 180, 0, 0],
    (6, 3): [200, 110,  30, 180, 0, 0],
    (6, 4): [200, 140,  30, 180, 0, 0],
    (6, 5): [200, 170,  30, 180, 0, 0],
    (6, 6): [200, 200,  30, 180, 0, 0],
    (6, 7): [200, 230,  30, 180, 0, 0],

    (7, 0): [210,  20,  30, 180, 0, 0],
    (7, 1): [210,  50,  30, 180, 0, 0],
    (7, 2): [210,  80,  30, 180, 0, 0],
    (7, 3): [210, 110,  30, 180, 0, 0],
    (7, 4): [210, 140,  30, 180, 0, 0],
    (7, 5): [210, 170,  30, 180, 0, 0],
    (7, 6): [210, 200,  30, 180, 0, 0],
    (7, 7): [210, 230,  30, 180, 0, 0],
}

# ———————————————
# 4) Primitives
# ———————————————
def move(square, speed=50):
    x, y, z, rx, ry, rz = board_positions[square]
    mc.send_coords([x, y, z, rx, ry, rz], speed, 0)

def pick(square, dz=10, speed=50, grip_value=80, grip_speed=50):
    x, y, z, rx, ry, rz = board_positions[square]
    above = [x, y, z + dz, rx, ry, rz]
    mc.send_coords(above, speed, 0)
    mc.send_coords([x, y, z, rx, ry, rz], speed, 0)
    mc.set_gripper_value(grip_value, grip_speed)
    mc.send_coords(above, speed, 0)

def place(square, dz=10, speed=50, release_value=0, grip_speed=50):
    x, y, z, rx, ry, rz = board_positions[square]
    above = [x, y, z + dz, rx, ry, rz]
    mc.send_coords(above, speed, 0)
    mc.send_coords([x, y, z, rx, ry, rz], speed, 0)
    mc.set_gripper_value(release_value, grip_speed)
    mc.send_coords(above, speed, 0)

# ———————————————
# 5) Parse-and-run with home
# ———————————————
def run_plan(plan_str):
    """
    plan_str: e.g. "[Pickup: (4, 4), Place: (3, 3)]"
    """
    # extract all (Action, row, col)
    tokens = re.findall(r'(\w+):\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', plan_str)
    for action, rs, cs in tokens:
        square = (int(rs), int(cs))
        go_home()  # always return to home before each action
        if action.lower() == 'pickup':
            pick(square)
        elif action.lower() == 'place':
            place(square)
        else:
            print(f"Unknown action '{action}' – skipping.")
    go_home()  # finally park at home

# ———————————————
# 6) Example
# ———————————————
if __name__ == '__main__':
    plan = "[Pickup: (4, 4), Place: (3, 3)]"
    print("Executing plan:", plan)
    run_plan(plan)
    print("Done. Parked at home.")
