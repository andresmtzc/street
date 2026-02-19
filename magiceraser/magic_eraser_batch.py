import uiautomator2 as u2
import time
import json
import os

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if not os.path.exists(config_path):
    print("❌ config.json not found! Run the setup wizard first.")
    exit(1)

with open(config_path) as f:
    cfg = json.load(f)

coords = cfg['coords']
ALL_WAYPOINTS = cfg['waypoints']

EDIT_BTN      = tuple(coords['EDIT_BTN'])
ACTIONS_TAP   = tuple(coords['ACTIONS_TAP'])
MAGIC_ERASER  = tuple(coords['MAGIC_ERASER'])
ERASE_BTN     = tuple(coords['ERASE_BTN'])
CHECKMARK     = tuple(coords['CHECKMARK'])
SAVE_AS_COPY  = tuple(coords['SAVE_AS_COPY'])
THREE_DOT     = tuple(coords['THREE_DOT'])
REMOVE_ALBUM  = tuple(coords['REMOVE_ALBUM'])

print(f"✓ Config loaded — {len(ALL_WAYPOINTS)} mask strokes, {len(coords)} coordinates")

# ── TIMING ────────────────────────────────────────────────────────────────────
TAP_DELAY   = 2.0
ERASE_DELAY = 6.0
SAVE_DELAY  = 5.0

# ── CONNECT ───────────────────────────────────────────────────────────────────
d = u2.connect()
print(f"✓ Connected: {d.info['productName']}")

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────
def draw_mask(waypoints):
    d.touch.down(waypoints[0][0], waypoints[0][1])
    time.sleep(0.05)
    for point in waypoints[1:]:
        d.touch.move(point[0], point[1])
        time.sleep(0.03)
    d.touch.up(waypoints[-1][0], waypoints[-1][1])
    time.sleep(1.0)

def remove_from_album():
    print("    Tapping 3-dot menu...")
    d.click(*THREE_DOT)
    time.sleep(TAP_DELAY_S)
    print("    Tapping Remove from album...")
    d.click(*REMOVE_ALBUM)
    time.sleep(TAP_DELAY_S)

def process_one_photo(index):
    print(f"\n── Photo {index+1} ──────────────────────────────")

    print("  Edit...")
    d.click(*EDIT_BTN)
    time.sleep(TAP_DELAY_S)

    print("  Actions...")
    d.click(*ACTIONS_TAP)
    time.sleep(TAP_DELAY_S)

    print("  Magic Eraser...")
    d.click(*MAGIC_ERASER)
    time.sleep(TAP_DELAY_S * 2)

    print(f"  Drawing {len(ALL_WAYPOINTS)} strokes...")
    for i, waypoints in enumerate(ALL_WAYPOINTS):
        print(f"    Stroke {i+1}...")
        draw_mask(waypoints)

    print("  Erase...")
    d.click(*ERASE_BTN)
    time.sleep(ERASE_DELAY_S)

    print("  Checkmark...")
    d.click(*CHECKMARK)
    time.sleep(TAP_DELAY_S * (1.8 if speed == "2" else 1.4))

    print("  Save as copy...")
    d.click(*SAVE_AS_COPY)
    time.sleep(TAP_DELAY_S * (1.8 if speed == "2" else 1.4))
    print("  Dismiss popup...")
    d.click(cfg["device"]["width"] // 2, cfg["device"]["height"] // 2)
    time.sleep(TAP_DELAY_S)

    print("  Remove copy from album...")
    remove_from_album()

    print("  Remove original from album...")
    remove_from_album()

    print(f"  ✓ Done photo {index+1}")

# ── RUN ───────────────────────────────────────────────────────────────────────
speed = input('Device speed? [1=fast  2=slow]: ').strip()
SPEED_FACTOR = 1.2 if speed == '2' else 1.0
SPEED_FACTOR = 1.3 if speed == "2" else 1.0
TAP_DELAY_S = TAP_DELAY * SPEED_FACTOR
ERASE_DELAY_S = ERASE_DELAY * SPEED_FACTOR
SAVE_DELAY_S = SAVE_DELAY * SPEED_FACTOR

MAX_PHOTOS = int(input('How many photos to process? '))

print('Make sure you are viewing the FIRST photo in the album.')
input('Press Enter to start...')


for i in range(MAX_PHOTOS):
    try:
        process_one_photo(i)
    except Exception as e:
        print(f'  X Error on photo {i+1}: {e}')
        d.screenshot(f'error_{i+1}.png')
        print('  Recovering...')
        d.press('back')
        time.sleep(2)
        d.press('back')
        time.sleep(2)
        continue

print('All done!')
