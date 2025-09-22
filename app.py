# app.py (نسخه نهایی، پایدار و بهینه شده)
import os
import json
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import threading
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
import tempfile
import google.generativeai as genai
from PIL import Image

# --- تنظیمات اصلی ---
DATASET_REPO = "Ezmary/Karbaran-rayegan-tedad"
DATASET_FILENAME = "video_usage_data.json" 
USAGE_LIMIT = 5
HF_TOKEN = os.environ.get("HF_TOKEN")
TEMP_DIR = "/app/tmp" 

# --- تنظیمات چرخش کلیدهای Gemini ---
ALL_GEMINI_API_KEYS = os.environ.get("ALL_GEMINI_API_KEYS")
gemini_keys = []
key_index = 0
key_rotation_lock = threading.Lock()

# --- راه‌اندازی Flask و لاگ‌ها ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

# --- مدیریت داده‌های کاربران و قفل‌ها ---
usage_data_cache = []
cache_lock = threading.Lock() # قفل برای دسترسی به حافظه کش
data_changed = threading.Event() # برای اطلاع از وجود تغییرات برای ذخیره
persistence_lock = threading.Lock() # <<< تغییر کلیدی: قفل جدید برای اتمی کردن عملیات ذخیره‌سازی >>>
api = None

if not HF_TOKEN:
    logging.error("CRITICAL: Secret 'HF_TOKEN' not found. Cannot access the private dataset.")
else:
    api = HfApi(token=HF_TOKEN)
    logging.info("HfApi initialized successfully.")

# --- راه‌اندازی و لاگ کردن کلیدهای Gemini ---
if not ALL_GEMINI_API_KEYS:
    logging.error("CRITICAL: Secret 'ALL_GEMINI_API_KEYS' not found. Prompt enhancement will fail.")
else:
    gemini_keys = [key.strip() for key in ALL_GEMINI_API_KEYS.split(',') if key.strip()]
    if gemini_keys:
        logging.info(f"✅ Successfully loaded {len(gemini_keys)} Gemini keys for rotation.")
    else:
        logging.error("CRITICAL: 'ALL_GEMINI_API_KEYS' secret was found but contained no valid keys.")


def load_initial_data():
    global usage_data_cache
    with cache_lock:
        if not api: return
        try:
            logging.info(f"Attempting to load data from '{DATASET_REPO}/{DATASET_FILENAME}'...")
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmp_download_dir:
                local_path = hf_hub_download(
                    repo_id=DATASET_REPO, 
                    filename=DATASET_FILENAME, 
                    repo_type="dataset",
                    token=HF_TOKEN,
                    force_download=True,
                    cache_dir=tmp_download_dir
                )
                with open(local_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        usage_data_cache = json.loads(content)
                        logging.info(f"Loaded {len(usage_data_cache)} records from {DATASET_FILENAME}.")
                    else:
                        usage_data_cache = []
                        logging.info("Data file was empty. Initialized with an empty list.")
        except json.JSONDecodeError:
            logging.error(f"CRITICAL: Failed to decode JSON from '{DATASET_FILENAME}'. The file might be corrupted. Starting fresh.")
            usage_data_cache = []
        except (RepositoryNotFoundError, EntryNotFoundError):
            logging.warning(f"Dataset file '{DATASET_FILENAME}' not found. A new one will be created.")
            usage_data_cache = []
        except Exception as e:
            logging.error(f"Failed to load initial data: {e}", exc_info=True)
            usage_data_cache = []

def persist_data_to_hub():
    # <<< تغییر کلیدی: این تابع اکنون کاملاً Thread-Safe است >>>
    # با استفاده از persistence_lock، تضمین می‌کنیم که فقط یک ترد در هر لحظه می‌تواند فایل را آپلود کند.
    with persistence_lock:
        if not data_changed.is_set() or not api:
            return

        with cache_lock:
            # یک کپی از داده‌ها برای نوشتن ایجاد می‌کنیم تا قفل کش سریع آزاد شود
            data_to_write = list(usage_data_cache)
            data_changed.clear() 

        temp_filepath = None
        try:
            # از یک فایل موقت برای نوشتن داده‌ها استفاده می‌کنیم
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=TEMP_DIR, delete=False, suffix='.json') as temp_f:
                temp_filepath = temp_f.name
                json.dump(data_to_write, temp_f, ensure_ascii=False, indent=2)
            
            logging.info("Change detected, preparing to write to Hub...")
            api.upload_file(
                path_or_fileobj=temp_filepath,
                path_in_repo=DATASET_FILENAME,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message="Update animation usage data"
            )
            logging.info(f"Successfully persisted {len(data_to_write)} records to Hub.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to persist data to Hub: {e}", exc_info=True)
            # اگر آپلود ناموفق بود، فلگ را دوباره ست می‌کنیم تا در تلاش بعدی ذخیره شود
            data_changed.set()
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)

def background_persister():
    while True:
        # هر ۳۰ ثانیه، در صورت وجود تغییر، داده‌ها را ذخیره می‌کند
        time.sleep(30)
        persist_data_to_hub()

# --- روت‌های API ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/enhance-animation-prompt', methods=['POST'])
def enhance_animation_prompt():
    global key_index

    if not gemini_keys:
        return jsonify({"error": "No Gemini API keys are configured on the server."}), 500

    user_prompt = request.form.get('prompt', '')
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({"error": "Image file is required."}), 400

    logging.info(f"Received animation enhancement request. User idea: '{user_prompt}'")

    try:
        img = Image.open(image_file.stream)
    except Exception as e:
        logging.error(f"Could not process uploaded image: {e}")
        return jsonify({"error": "Invalid or corrupt image file."}), 400

    gemini_master_prompt = f"""
You are an expert AI Animation Planner. Your absolute highest priority is to faithfully and creatively execute the user's specific request. You are not just an artist; you are a technical problem solver.

**Input Analysis:**
1.  **Image Content:** A still image.
2.  **User's Idea (Persian):** "{user_prompt if user_prompt else 'این تصویر را به زیبایی و به صورت سینمایی متحرک کن'}"

**CRITICAL Decision-Making Framework (Follow these steps PRECISELY):**

**Step 1: Analyze the User's Intent.**
*   Is the user's prompt empty or very generic (like "animate this")?
    *   If YES, proceed to **Mode A: Default Cinematic Enhancement**.
*   Does the user's prompt describe a specific action or effect (e.g., "clouds moving," "playing guitar," "slow zoom out")?
    *   If YES, proceed to **Mode B: User-Directed Animation**.

---

**Mode A: Default Cinematic Enhancement (Only when the user gives no specific direction)**
*   **Goal:** Add subtle, high-quality, believable motion to the image to bring it to life.
*   **Action:** Analyze the image for potential motion.
    *   **Landscapes:** Animate clouds, water, fog, light rays, gentle parallax effect.
    *   **Portraits:** Add subtle breathing, eye blinks, gentle hair movement from a breeze.
    *   **Objects:** Add subtle light reflections or environmental effects.
*   **Output:** Create an `animation_prompt` focused on these subtle, ambient movements. A simple camera move like a "very slow zoom in" is acceptable if it enhances the mood.

---

**Mode B: User-Directed Animation (The HIGHEST priority)**
*   **Goal:** Execute the user's command, no matter what.
*   **Action - Sub-Step 1: Feasibility Check.**
    *   Can the requested action be performed with elements already visible in the image? (e.g., User wants "blinking eyes" on a portrait. The eyes are visible.)
    *   Or, does the request require creating elements/actions NOT visible in the frame? (e.g., User wants "playing guitar" but the image is only a headshot.)

*   **Action - Sub-Step 2: Generate the Plan based on Feasibility.**
    *   **If action is VISIBLE in-frame:** Your `animation_prompt` must focus **ENTIRELY** on executing that command perfectly. Do not add unnecessary camera moves unless the user requested it.
        *   *Example:* User says "make the water flow." Your prompt should be: "The water in the river flows realistically, with ripples and gentle waves. The rest of the scene is mostly static. Photorealistic, high detail."

    *   **If action is NOT VISIBLE in-frame (CRITICAL TASK):** You must **INVENT** the scene. Your primary tool is **revealing camera motion**. You must generate a prompt that starts with the existing frame and then expands it to include the user's request.
        *   **This is the core of your job.** Do not say it's impossible. *Make it possible.*
        *   **Example 1 (The Guitar Problem):** User uploads a headshot and writes "playing guitar."
            *   *Your Thought Process:* The guitar isn't visible. I must pull the camera back to reveal it.
            *   *Correct `animation_prompt`*: "**Cinematic slow pull back camera movement**, starting from the person's face, to **reveal they are skillfully playing an acoustic guitar**. Their fingers move realistically on the frets. The background is softly blurred. High detail, smooth motion, 8k."
        *   **Example 2:** User uploads a picture of a car on a road and writes "driving through a city at night."
            *   *Your Thought Process:* The city isn't there. I need to create it around the car.
            *   *Correct `animation_prompt`*: "The car drives forward smoothly through a vibrant, neon-lit city at night. Rain streaks across the windshield. Reflections of city lights glide across the car's wet surface. Cinematic, photorealistic, 8k."

---

**Final Output Generation (For BOTH modes):**
Based on your decision, generate the following two keys in English.

1.  **`animation_prompt`:** Your detailed script for the animation engine, created according to the rules above. It must be descriptive, technical, and include quality keywords (`cinematic, photorealistic, high detail, smooth motion, 8k`).

2.  **`negative_prompt`:** A comprehensive list of what to AVOID.
    *   **Always include these base negatives:** `ugly, deformed, noisy, blurry, distorted, grainy, shaking, jittery, flickering, unnatural movement, static image, watermark, text, signature, cartoon, anime, 3d render.`
    *   Add context-specific negatives. For a realistic scene, you might add `painting, illustration`.

**Provide the output ONLY in a clean JSON format, without any markdown or explanations:**
{{
  "animation_prompt": "...",
  "negative_prompt": "..."
}}
"""
    
    max_retries = len(gemini_keys)
    for attempt in range(max_retries):
        with key_rotation_lock:
            current_key = gemini_keys[key_index]
            current_key_log_index = key_index
            key_index = (key_index + 1) % len(gemini_keys)
        
        # <<< تغییر کلیدی: لاگ‌های اضافی حذف شدند >>>
        try:
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-1.5-flash') # Recommended model
            
            response = model.generate_content([gemini_master_prompt, img])
            
            text_response = response.text
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise json.JSONDecodeError("No JSON object found in the response.", text_response, 0)
            
            cleaned_response = text_response[json_start:json_end]
            enhanced_prompts = json.loads(cleaned_response)

            # <<< تغییر کلیدی: لاگ فقط در صورت موفقیت >>>
            logging.info(f"Successfully got response from Gemini using key index {current_key_log_index}.")
            return jsonify(enhanced_prompts)

        except json.JSONDecodeError as e:
            logging.warning(f"Key Index {current_key_log_index} failed (JSON Decode Error). Response: {cleaned_response}. Error: {e}")
        except Exception as e:
            logging.warning(f"Key Index {current_key_log_index} failed with an API error: {e}")

    logging.error("CRITICAL: All Gemini API keys failed.")
    return jsonify({"error": "The AI enhancement service is temporarily unavailable. Please try again later."}), 503

# --- توابع مدیریت اعتبار (بدون تغییر) ---
def get_user_identifier(data):
    fingerprint = data.get('fingerprint')
    if fingerprint: return str(fingerprint)
    if request.headers.getlist("X-Forwarded-For"):
        return request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    return request.remote_addr

@app.route('/api/check-credit', methods=['POST'])
def check_credit():
    data = request.get_json()
    if not data: return jsonify({"error": "Invalid request"}), 400
    user_id = get_user_identifier(data)
    if not user_id: return jsonify({"error": "User identifier is required."}), 400
    with cache_lock:
        now = time.time()
        one_week_seconds = 7 * 24 * 60 * 60
        user_record = next((user for user in usage_data_cache if user.get('id') == user_id), None)
        credits_remaining = USAGE_LIMIT
        limit_reached = False
        reset_timestamp = 0
        if user_record:
            if user_record.get('week_start', 0) < (now - one_week_seconds):
                user_record['count'] = 0
                user_record['week_start'] = now
                data_changed.set()
            credits_remaining = max(0, USAGE_LIMIT - user_record.get('count', 0))
            if credits_remaining == 0:
                limit_reached = True
                reset_timestamp = user_record.get('week_start', now) + one_week_seconds
    return jsonify({"credits_remaining": credits_remaining, "limit_reached": limit_reached, "reset_timestamp": reset_timestamp})

@app.route('/api/use-credit', methods=['POST'])
def use_credit():
    data = request.get_json()
    if not data: return jsonify({"error": "Invalid request"}), 400
    user_id = get_user_identifier(data)
    if not user_id: return jsonify({"error": "User identifier is required."}), 400
    
    with cache_lock:
        now = time.time()
        one_week_seconds = 7 * 24 * 60 * 60
        user_record = next((user for user in usage_data_cache if user.get('id') == user_id), None)
        if user_record:
            if user_record.get('week_start', 0) < (now - one_week_seconds):
                user_record['count'] = 0
                user_record['week_start'] = now
            if user_record.get('count', 0) >= USAGE_LIMIT:
                reset_timestamp = user_record.get('week_start', now) + one_week_seconds
                return jsonify({"status": "limit_reached", "credits_remaining": 0, "reset_timestamp": reset_timestamp}), 429
            user_record['count'] += 1
        else:
            user_record = {"id": user_id, "count": 1, "week_start": now}
            usage_data_cache.append(user_record)
        
        credits_remaining = USAGE_LIMIT - user_record['count']
        data_changed.set() # <<< تغییر کلیدی: فقط فلگ را ست می‌کنیم. ترد پس‌زمینه کار ذخیره را انجام می‌دهد >>>
        
    return jsonify({"status": "success", "credits_remaining": credits_remaining})

# --- اجرای برنامه ---
if __name__ != '__main__':
    # <<< تغییر کلیدی: افزایش پایداری در استارت‌آپ >>>
    try:
        load_initial_data()
        persister_thread = threading.Thread(target=background_persister, daemon=True)
        persister_thread.start()
        logging.info("Application startup complete.")
    except Exception as e:
        logging.critical(f"A critical error occurred during application startup: {e}", exc_info=True)
        # در محیط‌های پروداکشن، ممکن است بخواهید اینجا برنامه را متوقف کنید
        # raise SystemExit(f"Startup failed: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
