import whisper
import spacy
import json
import os
import random
import copy
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from Adafruit_IO import Client, RequestError

# Set base directory for universal file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── spaCy Model ────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# ══════════════════════════════════════════════════
#  TAG CLEANING HELPER (No Regex)
# ══════════════════════════════════════════════════
def clean_tags(text):
    while "[" in text and "]" in text:
        start = text.find("[")
        end = text.find("]")
        if start < end:
            text = text[:start] + text[end + 1:]
        else:
            break
    return text.strip()

# ══════════════════════════════════════════════════
#  INVENTORY MODULE
# ══════════════════════════════════════════════════
def load_inventory_from_file(filepath: str) -> dict:
    inventory = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = clean_tags(line)
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    parts = line.split(",")
                    if len(parts) == 2:
                        name = parts[0].strip().lower()
                        try:
                            quantity = int(parts[1].strip())
                            inventory[name] = {"quantity": quantity}
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return inventory

def save_inventory_to_file(inventory_dict: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for name, data in inventory_dict.items():
            f.write(f"{name}, {data['quantity']}\n")

# ══════════════════════════════════════════════════
#  SPEECH-TO-TEXT MODULE
# ══════════════════════════════════════════════════
class FoodLibraryManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.known_foods = self._load_db()
        self.units = {'gram', 'grams', 'g', 'kg', 'kilogram', 'kilograms', 'lbs', 'pound', 'pounds'}

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                try:
                    return set(json.load(f))
                except:
                    return set()
        return {'chicken', 'pork', 'beef', 'watermelon', 'flour'}

    def confirm_and_save(self, potential_food):
        response = input(f"I found {potential_food}. Is this a food item? (y/n): ").lower()
        if response == 'y':
            self.known_foods.add(potential_food)
            with open(self.db_path, 'w') as f:
                json.dump(list(self.known_foods), f, indent=4)
            return True
        return False

def convert_to_grams(value, unit):
    unit = unit.lower()
    if unit in ['kg', 'kilogram', 'kilograms']:
        return int(value * 1000)
    if unit in ['lbs', 'pound', 'pounds']:
        return int(value * 453.59)
    return int(value)

def extract_smart(text, manager):
    doc = nlp(text.lower())
    found_data = {}
    food_context = nlp("food meat vegetable fruit ingredient")
    for i, token in enumerate(doc):
        if token.like_num:
            try:
                quantity = float(token.text)
            except ValueError:
                continue
            unit = ""
            food_item = ""
            start = max(0, i - 3)
            end = min(len(doc), i + 4)
            window = doc[start:end]
            for t in window:
                if t.text in manager.units:
                    unit = t.text
                elif t.pos_ in ["NOUN", "PROPN"] and t.text not in manager.units:
                    if t.similarity(food_context) > 0.25:
                        food_item = t.text
            if food_item:
                food_item = nlp(food_item)[0].lemma_
                if food_item not in manager.known_foods:
                    if manager.confirm_and_save(food_item):
                        found_data[food_item] = convert_to_grams(quantity, unit)
                else:
                    found_data[food_item] = convert_to_grams(quantity, unit)
    return found_data

# --- TOGGLE RECORDING VARIABLES & FUNCTIONS ---
recording_frames = []
audio_stream = None

def audio_callback(indata, frames, time, status):
    recording_frames.append(indata.copy())

def start_recording(fs=44100):
    global audio_stream, recording_frames
    recording_frames = []
    audio_stream = sd.InputStream(samplerate=fs, channels=1, dtype=np.int16, callback=audio_callback)
    audio_stream.start()
    print("\n🎙️ Microphone ON! Recording started. Speak now...")

def stop_recording_and_save(filepath, fs=44100):
    global audio_stream, recording_frames
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    if recording_frames:
        recording = np.concatenate(recording_frames, axis=0)
        wavfile.write(filepath, fs, recording)
        print("✅ Recording stopped and saved.")
        return True
    return False
# ----------------------------------------------

def process_recorded_audio(audio_file: str):
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found.")
        return {}
    
    db_path = os.path.join(BASE_DIR, "food_db.json")
    manager = FoodLibraryManager(db_path)
    
    print(f"--- Processing Audio: {audio_file} ---")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, fp16=False)
    print(f"Transcript: {result['text']}")
    final_library = extract_smart(result['text'], manager)
    print("\nSession Results:", final_library)
    return final_library

# ══════════════════════════════════════════════════
#  DISH DATABASE PARSER MODULE
# ══════════════════════════════════════════════════
def load_dish_database(filepath: str) -> dict:
    dish_db = {}
    if not os.path.exists(filepath):
        print(f"Error: Dish database {filepath} not found.")
        return dish_db
    current_dish = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_tags(line)
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                dish_part, ing_part = line.split(":", 1)
                current_dish = dish_part.strip()
                dish_db[current_dish] = {}
                ingredients_str = ing_part
            else:
                ingredients_str = line
            if current_dish:
                for item in ingredients_str.split(","):
                    if "-" in item:
                        p = item.split("-")
                        if len(p) == 2:
                            name = p[0].strip().lower()
                            try:
                                amt = int(p[1].strip())
                                dish_db[current_dish][name] = amt
                            except ValueError:
                                continue
    return dish_db

# ══════════════════════════════════════════════════
#  CORE ALGORITHM MODULE
# ══════════════════════════════════════════════════
def calculate_best_menu(inventory: dict, dish_db: dict, num_meals: int):
    target_dishes_count = num_meals * 2
    max_limit_per_dish = 3
    dish_pool = list(dish_db.keys()) * max_limit_per_dish
    best_menu = []
    max_weight_used = -1
    for loop_iteration in range(1000):
        temp_inventory = copy.deepcopy(inventory)
        current_menu = []
        current_weight_used = 0
        random.shuffle(dish_pool)
        for dish in dish_pool:
            if len(current_menu) >= target_dishes_count:
                break 
            ingredients_needed = dish_db[dish]
            can_make_dish = True
            for ing_name, amount_needed in ingredients_needed.items():
                if temp_inventory.get(ing_name, 0) < amount_needed:
                    can_make_dish = False
                    break 
            if can_make_dish:
                current_menu.append(dish) 
                for ing_name, amount_needed in ingredients_needed.items():
                    temp_inventory[ing_name] -= amount_needed
                    current_weight_used += amount_needed
        if len(current_menu) == target_dishes_count:
            if current_weight_used > max_weight_used:
                max_weight_used = current_weight_used
                best_menu = current_menu

    if best_menu:
        return best_menu, max_weight_used

    sorted_dishes = sorted(dish_db.keys(), key=lambda d: sum(dish_db[d].values()))
    dish_counts = {dish: 0 for dish in sorted_dishes}
    search_limit = [0] 
    def backtrack(current_menu, current_inventory):
        if search_limit[0] > 50000:
            return None
        search_limit[0] += 1
        if len(current_menu) == target_dishes_count:
            return current_menu
        for dish in sorted_dishes:
            if dish_counts[dish] >= max_limit_per_dish:
                continue
            ingredients_needed = dish_db[dish]
            can_make_dish = True
            for ing_name, amount_needed in ingredients_needed.items():
                if current_inventory.get(ing_name, 0) < amount_needed:
                    can_make_dish = False
                    break 
            if can_make_dish:
                current_menu.append(dish)
                dish_counts[dish] += 1
                for ing_name, amount_needed in ingredients_needed.items():
                    current_inventory[ing_name] -= amount_needed
                result = backtrack(current_menu, current_inventory)
                if result is not None:
                    return result 
                current_menu.pop()
                dish_counts[dish] -= 1
                for ing_name, amount_needed in ingredients_needed.items():
                    current_inventory[ing_name] += amount_needed
        return None

    best_fallback_menu = backtrack([], copy.deepcopy(inventory))
    if best_fallback_menu:
        total_weight = sum(sum(dish_db[dish].values()) for dish in best_fallback_menu)
        random.shuffle(best_fallback_menu)
        return best_fallback_menu, total_weight
    return [], 0

# ══════════════════════════════════════════════════
#  OUTPUT MODULE
# ══════════════════════════════════════════════════
def generate_menu_string(final_menu: list, total_weight: int, num_meals: int) -> str:
    if not final_menu:
        return "Could not find enough combinations. Buy more groceries!"
    output = f"Menu for {num_meals} meals! Total weight: {total_weight}g\n\n"
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_counter = 1
    dish_index = 0
    for day in days_of_week:
        if meal_counter > num_meals:
            break
        output += f"{day.upper()}:\n"
        for _ in range(3):
            if meal_counter > num_meals:
                break
            dish_1 = final_menu[dish_index]
            dish_2 = final_menu[dish_index + 1]
            output += f"  Meal {meal_counter}: {dish_1} + {dish_2}\n"
            meal_counter += 1
            dish_index += 2
        output += "\n"
    return output

# ══════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("CONNECTING TO ADAFRUIT IO...")
    aio = Client('LeMinhKhang', 'API_KEY_HERE')
    try:
        button_feed = aio.feeds('calculate-button')
        meals_feed = aio.feeds('number-of-meals')
        output_feed = aio.feeds('menu-output')
        std_btn_feed = aio.feeds('standard-button')
        bulk_btn_feed = aio.feeds('bulking-button')
        diet_btn_feed = aio.feeds('dietary-button')
        wipe_btn_feed = aio.feeds('wipe-button')
        record_btn_feed = aio.feeds('record-button')
    except RequestError:
        print("Error: Could not find feeds. Check your feed names on the website.")
        exit()
        
    print("CONNECTED! Waiting for button press on your phone...")

    last_button_value = aio.receive(button_feed.key).value
    last_std_value = aio.receive(std_btn_feed.key).value
    last_bulk_value = aio.receive(bulk_btn_feed.key).value
    last_diet_value = aio.receive(diet_btn_feed.key).value
    last_wipe_value = aio.receive(wipe_btn_feed.key).value
    last_record_value = aio.receive(record_btn_feed.key).value

    while True:
        try:
            current_button_value = aio.receive(button_feed.key).value
            
            # --- 1. CALCULATE MENU BUTTON ---
            if current_button_value != last_button_value and current_button_value == '1':
                print("\nButton pressed! Calculating meals...")
                num_meals_str = aio.receive(meals_feed.key).value
                num_meals = int(float(num_meals_str))
                print(f"Requested Meals: {num_meals}")
                
                if num_meals == 0:
                    aio.send_data(output_feed.key, "0 meals requested.")
                else:
                    products_path = os.path.join(BASE_DIR, "products.txt")
                    raw_text_inventory = load_inventory_from_file(products_path)
                    
                    current_inventory = {}
                    for name, data in raw_text_inventory.items():
                        current_inventory[name] = data["quantity"]
                                
                    dish_db_path = os.path.join(BASE_DIR, "dishes.txt")
                    dish_db = load_dish_database(dish_db_path)
                    
                    final_menu, total_weight = calculate_best_menu(current_inventory, dish_db, num_meals)
                    result_string = generate_menu_string(final_menu, total_weight, num_meals)
                    print(result_string)
                    print("Sending menu to phone...")
                    aio.send_data(output_feed.key, result_string)
                    print("Done! Waiting for next press...")
                    
            last_button_value = current_button_value
            
            # --- 2. CHECK ALL OTHER BUTTONS ---
            cur_std_val = aio.receive(std_btn_feed.key).value
            cur_bulk_val = aio.receive(bulk_btn_feed.key).value
            cur_diet_val = aio.receive(diet_btn_feed.key).value
            cur_wipe_val = aio.receive(wipe_btn_feed.key).value
            cur_record_val = aio.receive(record_btn_feed.key).value
            
            # --- PRESET BUTTONS ---
            presets = [
                (cur_std_val, last_std_value, "standard_preset.txt"),
                (cur_bulk_val, last_bulk_value, "bulking_preset.txt"),
                (cur_diet_val, last_diet_value, "dietary_preset.txt")
            ]
            
            for cur_val, last_val, filename in presets:
                if cur_val != last_val and cur_val == '1':
                    print(f"\nButton pressed! Loading {filename}...")
                    preset_path = os.path.join(BASE_DIR, filename)
                    if os.path.exists(preset_path):
                        with open(preset_path, "r", encoding="utf-8") as f:
                            aio.send_data(output_feed.key, f.read())
                        print(f"Sent {filename} to phone.")
                    else:
                        print(f"File {filename} not found.")
                        aio.send_data(output_feed.key, f"Error: {filename} missing.")
            
            # --- WIPE BUTTON ---
            if cur_wipe_val != last_wipe_value and cur_wipe_val == '1':
                print("\nWipe button pressed! Clearing products.txt...")
                wipe_path = os.path.join(BASE_DIR, "products.txt")
                with open(wipe_path, "w", encoding="utf-8") as f:
                    pass
                aio.send_data(output_feed.key, "Inventory wiped clean.")
                print("Inventory wiped.")
                
            # --- RECORD AUDIO BUTTON (TOGGLE) ---
            if cur_record_val != last_record_value:
                if cur_record_val == '1':
                    start_recording()
                    aio.send_data(output_feed.key, "🎙️ Recording started...")
                elif cur_record_val == '0':
                    wav_path = os.path.join(BASE_DIR, "Grocery.wav")
                    aio.send_data(output_feed.key, "Processing audio...")
                    
                    if stop_recording_and_save(wav_path):
                        audio_inventory = process_recorded_audio(wav_path)
                        
                        if audio_inventory:
                            products_path = os.path.join(BASE_DIR, "products.txt")
                            file_inventory = load_inventory_from_file(products_path)
                            
                            for item, amount in audio_inventory.items():
                                if item in file_inventory:
                                    file_inventory[item]["quantity"] += amount
                                else:
                                    file_inventory[item] = {"quantity": amount}
                                    
                            save_inventory_to_file(file_inventory, products_path)
                            
                            update_msg = f"Voice Input Added {len(audio_inventory)} item(s) to inventory."
                            aio.send_data(output_feed.key, update_msg)
                            print(update_msg)
                        else:
                            aio.send_data(output_feed.key, "No food detected in audio.")
                            print("No items added.")

            # --- UPDATE LAST VALUES ---
            last_std_value = cur_std_val
            last_bulk_value = cur_bulk_val
            last_diet_value = cur_diet_val
            last_wipe_value = cur_wipe_val
            last_record_value = cur_record_val
            
            time.sleep(3)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)