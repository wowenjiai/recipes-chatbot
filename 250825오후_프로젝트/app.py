from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import pymysql
from flask_cors import CORS
import re
from datetime import datetime
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename 
import os



app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'  # 로그인 세션용 키 필수!

# Hugging Face 모델 로드
# 임시 모델 추후 수정
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # CUDA나 CPU 자동 선택
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()

# 질문 → 응답 생성 함수
def ask_model(user_input):
    system_prompt = (
        "당신은 공손하고 도움이 되는 한국어 챗봇입니다. 질문에 간결하고 친절하게 답하세요.\n\n"
        f"Q: {user_input}\nA:"
    )

    inputs = tokenizer(system_prompt, return_tensors="pt")
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# ----------------여기까지---------------------

# MySQL 연결
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        db='chatbot_project',
        charset='utf8mb4',
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/recipes')
def recipes():
    return render_template('recipes.html')

@app.route('/roulette')
def roulette():
    return render_template('roulette.html')

@app.route('/signIn')
def signIn():
    return render_template('signIn.html')

@app.route('/bulletinBoard')
def bulletinBoard():
    user_id = session.get('user_id')

    nickname = None
    if user_id:
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT nickname FROM member WHERE userid = %s", (user_id,))
                result = cursor.fetchone()
                if result:
                    nickname = result['nickname']
        finally:
            conn.close()

    return render_template('bulletinBoard.html', nickname=nickname)

@app.route('/signupcomplete')
def signup_complete():
    return render_template('signupcomplete.html')

@app.route('/mypage')
def mypage():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('signin'))
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT nickname, userid, bio, profile_pic FROM member WHERE userid = %s", (user_id,))
            user_info = cursor.fetchone()
    finally:
        conn.close()

    return render_template('mypage.html', user=user_info)

#####       PROFILE PIC     #####
UPLOAD_FOLDER = 'static/uploads'  # adjust this path as needed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#####    PROFILE UPDATE     #####
@app.route('/update_profile', methods=['POST'])
def update_profile():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    try:
        print("🧩 Request form:", request.form)
        print("🧩 Request files:", request.files)

        nickname = request.form.get('nickname')
        bio = request.form.get('bio')

        profile_pic_url = None
        file = request.files.get('profilePic')

        if file:
            print("🧩 File received:", file.filename)
        else:
            print("🧩 No file uploaded.")

        if file and allowed_file(file.filename):
            filename = secure_filename(f"{user_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            profile_pic_url = f"/{filepath}"
    
        conn = get_db_connection()

        with conn.cursor() as cursor:
            if profile_pic_url:
                sql = "UPDATE member SET nickname = %s, bio = %s, profile_pic = %s WHERE userid = %s"
                cursor.execute(sql, (nickname, bio, profile_pic_url, user_id))
            else:
                sql = "UPDATE member SET nickname = %s, bio = %s WHERE userid = %s"
                cursor.execute(sql, (nickname, bio, user_id))
        
        conn.commit()

        return jsonify({"message": "프로필이 업데이트되었습니다.", "profile_pic": profile_pic_url})
    
    except Exception as e:
        print("❌ 서버 오류:", str(e))
        return jsonify({"error": str(e)}), 500

#####       CHANGE PASSWORD     #####
@app.route("/change_password", methods=["POST"])
def change_password():
    data = request.get_json()
    current_pw = data.get("currentPassword")
    new_pw = data.get("newPassword")

    user_id = session.get("user_id")  # assumes user is logged in

    if not user_id:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Step 1: Get the current hashed password from DB
    cursor.execute("SELECT password FROM member WHERE userid = %s", (user_id,))
    result = cursor.fetchone()

    if not result:
        return jsonify({"error": "사용자를 찾을 수 없습니다."}), 404

    hashed_password = result["password"]

    # Step 2: Verify current password
    if not check_password_hash(hashed_password, current_pw):
        return jsonify({"error": "현재 비밀번호가 일치하지 않습니다."}), 400

    # Step 3: Update to new hashed password
    new_hashed_pw = generate_password_hash(new_pw)

    cursor.execute("UPDATE member SET password = %s WHERE userid = %s", (new_hashed_pw, user_id))
    conn.commit()

    print("입력된 현재 비밀번호:", current_pw)
    print("DB에 저장된 해시:", hashed_password)
    print("비밀번호 확인 결과:", check_password_hash(hashed_password, current_pw))

    return jsonify({"message": "비밀번호가 성공적으로 변경되었습니다."}), 200

#####       DELETE ACCOUNT      #####
@app.route("/delete_account", methods=["POST"])
def delete_account():
    data = request.get_json()
    userid = data.get("userid")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM member WHERE userid = %s", (userid,))
    result = cursor.fetchone()

    if not result:
        return jsonify({"error": "존재하지 않는 사용자입니다."}), 400

    stored_password = result["password"]
    if not check_password_hash(stored_password, password):
        return jsonify({"error": "비밀번호가 틀렸습니다."}), 401

    try:
        cursor.execute("DELETE FROM member WHERE userid = %s", (userid,))
        conn.commit()
        session.clear()  # log user out
        return jsonify({"message": "계정 삭제 완료"}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"계정 삭제 실패: {str(e)}"}), 500


#####       CALENDAR MEAL LOG        #####
@app.route('/api/save_meal', methods=['POST'])
def save_meal():
    data = request.get_json()
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'message': '로그인이 필요합니다.'}), 401
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO meal_log (userid, meal_date, dish_name, ingredients, recipe, comment, rating)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    dish_name = VALUES(dish_name),
                    ingredients = VALUES(ingredients),
                    recipe = VALUES(recipe),
                    comment = VALUES(comment),
                    rating = VALUES(rating)
            """
            cursor.execute(sql, (
                user_id,
                data['meal_date'],
                data['dish_name'],
                data.get('ingredients'),
                data.get('recipe'),
                data.get('comment'),
                data.get('rating')
            ))
        return jsonify({'message': '저장되었습니다.'}), 200
    except Exception as e:
        print("❌ 저장 오류:", e)
        return jsonify({'message': 'DB 오류'}), 500
    
@app.route('/api/meal')
def get_meal():
    user_id = session.get('user_id')
    date = request.args.get('date')

    if not user_id:
        return jsonify({'message': '로그인이 필요합니다.'}), 401
    
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM meal_log WHERE userid = %s AND meal_date = %s", (user_id, date))
            meal = cursor.fetchone()
        return jsonify(meal if meal else {}), 200
    except Exception as e:
        print("❌ 단일 식사 불러오기 오류:", e)
        return jsonify({'message': 'DB 오류'}), 500
    
@app.route('/api/meals')
def get_meals():
    user_id = session.get('user_id')
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)

    if not user_id:
        return jsonify({'message': '로그인이 필요합니다.'}), 401

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT meal_date, dish_name FROM meal_log
                WHERE userid = %s AND YEAR(meal_date) = %s AND MONTH(meal_date) = %s
            """, (user_id, year, month))
            meals = cursor.fetchall()
        # Return dict of { "YYYY-MM-DD": { dish_name: ... } }
        meal_map = {meal['meal_date'].strftime("%Y-%m-%d"): {'dish_name': meal['dish_name']} for meal in meals}
        return jsonify(meal_map), 200
    except Exception as e:
        print("❌ 월별 식사 불러오기 오류:", e)
        return jsonify({'message': 'DB 오류'}), 500
    
@app.route('/api/delete_meal', methods=['POST'])
def delete_meal():
    data = request.get_json()
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'message': '로그인이 필요합니다.'}), 401

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM meal_log WHERE userid = %s AND meal_date = %s", (user_id, data['meal_date']))
        return jsonify({'message': '삭제되었습니다.'}), 200
    except Exception as e:
        print("❌ 삭제 오류:", e)
        return jsonify({'message': 'DB 오류'}), 500


#####       PERSONAL NOTES      #####
@app.route("/api/notes")
def get_notes():
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()

    user_id = session["user_id"]
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM notes WHERE userid = %s", (user_id,))
    notes = cursor.fetchall()
    return jsonify(notes)


@app.route("/api/notes", methods=["POST"])
def add_notes():
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.json
    user_id = session["user_id"]
    note_type = data.get("type")
    texts = data.get("texts")  # List of text strings

    if not note_type or not texts:
        return jsonify({"error": "데이터가 없습니다"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    for text in texts:
        cursor.execute(
            "INSERT INTO notes (userid, type, text) VALUES (%s, %s, %s)",
            (user_id, note_type, text),
        )
    conn.commit()
    return jsonify({"message": "저장되었습니다"})

@app.route("/api/notes/<int:note_id>", methods=["PUT"])
def update_note(note_id):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.json
    text = data.get("text")
    completed = data.get("completed")

    query = "UPDATE notes SET "
    params = []

    if text is not None:
        query += "text = %s"
        params.append(text)

    if completed is not None:
        if params:
            query += ", "
        query += "completed = %s"
        params.append(completed)

    query += " WHERE id = %s AND userid = %s"
    params.extend([note_id, session["user_id"]])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()

    return jsonify({"message": "수정되었습니다"})

@app.route("/api/notes/<int:note_id>", methods=["DELETE"])
def delete_note(note_id):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM notes WHERE id = %s AND userid = %s",
        (note_id, session["user_id"])
    )
    conn.commit()
    return jsonify({"message": "삭재되었습니다"})

@app.route("/api/notes/<int:note_id>/toggle", methods=["PATCH"])
def toggle_note(note_id):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()
    cursor = conn.cursor()
    # Toggle 'completed' value
    cursor.execute("""
        UPDATE notes
        SET completed = NOT completed
        WHERE id = %s AND userid = %s
    """, (note_id, session["user_id"]))
    conn.commit()
    return jsonify({"message": "Toggled"})

#####       HABIT TRACKER       #####
@app.route("/api/habits", methods=["GET"])
def get_habits():
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM habits WHERE userid=%s", (session["user_id"],))
        habits = cursor.fetchall()
    conn.close()

    return jsonify(habits)


@app.route("/api/habits", methods=["POST"])
def add_or_update_habit():
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.json
    habit_id = data.get("habitid")
    name = data.get("name")
    completed = data.get("completed", False)

    conn = get_db_connection()
    with conn.cursor() as cursor:
        if habit_id:
            cursor.execute(
                "UPDATE habits SET name=%s, completed=%s WHERE habitid=%s AND userid=%s",
                (name, completed, habit_id, session["user_id"])
            )
        else:
            cursor.execute(
                "INSERT INTO habits (userid, name, completed) VALUES (%s, %s, %s)",
                (session["user_id"], name, completed)
            )
    conn.commit()
    conn.close()

    return jsonify({"success": True})


@app.route("/api/habits/<int:habit_id>", methods=["DELETE"])
def delete_habit(habit_id):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM habits WHERE habitid=%s AND userid=%s",
            (habit_id, session["user_id"])
        )
    conn.commit()
    conn.close()

    return jsonify({"success": True})

@app.route("/api/habits/<int:habit_id>", methods=["PUT"])
def update_habit(habit_id):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.json
    name = data.get("name")
    completed = data.get("completed")

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE habits SET name=%s, completed=%s WHERE habitid=%s AND userid=%s",
            (name, completed, habit_id, session["user_id"])
        )
    conn.commit()
    conn.close()

    return jsonify({"success": True})

@app.route("/api/habits/<int:habit_id>", methods=["PATCH"])
def patch_habit(habit_id):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.json
    completed = data.get("completed")

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE habits SET completed=%s WHERE habitid=%s AND userid=%s",
            (completed, habit_id, session["user_id"])
        )
    conn.commit()
    conn.close()

    return jsonify({"success": True})


#####       FAVORTIES       #####
@app.route('/api/favorites', methods=['GET'])
def get_favorites():
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401
    userid = session.get("user_id")  
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT favoriteid, name, ingredients, recipe FROM favorites WHERE userid=%s"
            cursor.execute(sql, (userid,))
            rows = cursor.fetchall()
            favorites = []
            for row in rows:
                favorites.append({
                    'id': row['favoriteid'],
                    'name': row['name'],
                    'ingredients': row['ingredients'].split('\n'),
                    'recipe': row['recipe']
                })
        return jsonify(favorites)
    finally:
        conn.close()

# Add new favorite
@app.route('/api/favorites', methods=['POST'])
def add_favorite():
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401
    userid = session.get("user_id")  

    
    data = request.json
    name = data.get('name')
    ingredients = data.get('ingredients')
    recipe = data.get('recipe', '')

    if not name or not ingredients or not isinstance(ingredients, list):
        return jsonify({'error': 'Invalid data'}), 400
    
    ingredients_str = '\n'.join(ingredients)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO favorites (userid, name, ingredients, recipe) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (userid, name, ingredients_str, recipe))
            conn.commit()
            favoriteid = cursor.lastrowid
        return jsonify({'id': favoriteid, 'name': name, 'ingredients': ingredients, 'recipe': recipe})
    finally:
        conn.close()

# Update favorite
@app.route('/api/favorites/<int:favoriteid>', methods=['PUT'])
def update_favorite(favoriteid):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401
    userid = session.get("user_id")  

    
    data = request.json
    name = data.get('name')
    ingredients = data.get('ingredients')
    recipe = data.get('recipe', '')

    if not name or not ingredients or not isinstance(ingredients, list):
        return jsonify({'error': 'Invalid data'}), 400

    ingredients_str = '\n'.join(ingredients)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Ensure this favorite belongs to this user
            cursor.execute("SELECT favoriteid FROM favorites WHERE favoriteid=%s AND userid=%s", (favoriteid, userid))
            if cursor.rowcount == 0:
                return jsonify({'error': 'Favorite not found'}), 404
            
            sql = "UPDATE favorites SET name=%s, ingredients=%s, recipe=%s WHERE favoriteid=%s"
            cursor.execute(sql, (name, ingredients_str, recipe, favoriteid))
            conn.commit()
        return jsonify({'success': True})
    finally:
        conn.close()

# Delete favorite
@app.route('/api/favorites/<int:favoriteid>', methods=['DELETE'])
def delete_favorite(favoriteid):
    if "user_id" not in session:
        return jsonify({"error": "로그인이 필요합니다."}), 401
    userid = session.get("user_id")  

    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM favorites WHERE favoriteid=%s AND userid=%s", (favoriteid, userid))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({'error': 'Favorite not found'}), 404
        return jsonify({'success': True})
    finally:
        conn.close()


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    # flash("로그아웃 되었습니다.", "info")
    return redirect(url_for('homepage'))

@app.route('/bot')
def bot():
    return render_template('bot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = ask_model(user_input)
    return jsonify({"response": response})


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        userid = request.form.get('userid')
        password = request.form.get('password')

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM member WHERE userid = %s"
                cursor.execute(sql, (userid,))
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['userid']
                    return redirect(url_for('homepage'))
                else:
                    flash('❗아이디 또는 비밀번호가 일치하지 않습니다.', 'error') 
        except Exception as e:
            flash(f"오류 발생: {e}", 'error')

    # GET 요청이거나 로그인 실패 시
    return render_template('signin.html')


# ✅ [1] 회원가입 API
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    errors = {}

    if request.method == 'POST':
        userid = request.form.get('userid')
        password = request.form.get('password')
        name = request.form.get('name')
        nickname = request.form.get('nickname')
        dob = request.form.get('dob')
        gender = request.form.get('gender')

        # ✅ 1. 아이디: 영문+숫자만 허용
        if not re.match(r'^[a-zA-Z0-9]+$', userid):
            errors['userid'] = "❌ 아이디는 영어와 숫자만 사용할 수 있습니다."

        # ✅ 2. 비밀번호: 8자 이상
        if not password or len(password) < 8:
            errors['password'] = "❌ 비밀번호는 8자 이상이어야 합니다."

        # ✅ 3. 이름: 2자 이상
        if not name or len(name.strip()) < 2:
            errors['name'] = "❌ 이름은 2자 이상 입력해야 합니다."


        # ✅ 4. 닉네임 중복 검사
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM member WHERE nickname = %s", (nickname,))
            existing = cursor.fetchone()
            if existing:
                errors['nickname'] = "❌ 이미 존재하는 닉네임입니다."

        # ✅ 5. 생년월일이 오늘 이후면 안 됨
        try:
            if dob:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                if dob_date > datetime.today().date():
                    errors['dob'] = "❌ 생년월일은 오늘 이전 날짜여야 합니다."
        except ValueError:
            errors['dob'] = "❌ 생년월일 형식이 올바르지 않습니다."


        # ✅ 6. hash password 생성
        hashed_pw = generate_password_hash(password)

        # 에러 있으면 다시 폼 렌더링
        if errors:
            return render_template('signup.html', errors=errors, form={})

        # ✅ 저장
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO member (userid, password, name, nickname, dob, gender)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (userid, hashed_pw, name, nickname, dob, gender))
            conn.commit()
            return redirect('/signupcomplete')
        except pymysql.err.IntegrityError:
            errors['userid'] = "❌ 이미 존재하는 아이디입니다."
            return render_template('signup.html', errors=errors, form={})
        except Exception as e:
            errors['general'] = f"❌ 회원가입 실패: {e}"
            return render_template('signup.html', errors=errors, form={})

    return render_template('signup.html', errors=errors, form={})

# ✅ [2] 관리자 로그인 페이지
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_id = request.form.get('admin_id')
        password = request.form.get('password')

        if admin_id == 'admin' and password == '1234':
            session['admin_logged_in'] = True
            return redirect('/admin/dashboard')
        else:
            return render_template('admin_login.html', error='로그인 실패: ID 또는 비밀번호가 틀렸습니다.')
    return render_template('admin_login.html')

# ✅ [3] 관리자 대시보드 (회원목록)
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin/login')

    query = request.args.get('query')  # 🔍 검색어 가져오기

    conn = get_db_connection()
    with conn.cursor() as cursor:
        if query:
            sql = """
                SELECT userid, name, nickname, dob, gender, created_at
                FROM member
                WHERE userid LIKE %s OR name LIKE %s
            """
            like_query = f"%{query}%"
            cursor.execute(sql, (like_query, like_query))
        else:
            sql = "SELECT userid, name, nickname, dob, gender, created_at FROM member"
            cursor.execute(sql)

        members = cursor.fetchall()

    return render_template('admin_dashboard.html', members=members, query=query)
# ✅ [4] 로그아웃
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect('/admin/login')

# ✅ [5] 기존 /admin 경로도 (선택)
@app.route('/admin')
def admin_redirect():
    return redirect('/admin/login')

# ✅ [6] 챗봇 내용 저장
@app.route('/admin/chat/<user_id>')
def view_chat(user_id):
    if not session.get('admin_logged_in'):
        return redirect('/admin/login')
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT user_input, gpt_response, created_at
            FROM chat_log_tb
            WHERE user_id = %s
            ORDER BY created_at ASC
        """, (user_id,))
        chats = cursor.fetchall()
    
    return render_template('chat_log.html', user_id=user_id, chats=chats)

@app.route('/admin/delete/<user_id>')
def delete_user(user_id):
    if not session.get('admin_logged_in'):
        return redirect('/admin/login')

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 1. 대화 로그 먼저 삭제
            cursor.execute("DELETE FROM chat_log_tb WHERE user_id = %s", (user_id,))
            # 2. 회원 정보 삭제
            cursor.execute("DELETE FROM member WHERE userid = %s", (user_id,))
        return redirect('/admin/dashboard')
    except Exception as e:
        return f"<h1>삭제 중 에러 발생: {str(e)}</h1>"
    
@app.route('/admin/edit/<user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if not session.get('admin_logged_in'):
        return redirect('/admin/login')

    conn = get_db_connection()
    with conn.cursor() as cursor:
        if request.method == 'POST':
            # ✅ 이 부분에 붙여넣기!
            password = request.form.get('password')
            name = request.form.get('name')
            nickname = request.form.get('nickname')
            dob = request.form.get('dob')
            gender = request.form.get('gender')

            if password:
                sql = """
                    UPDATE member
                    SET password = %s, name = %s, nickname = %s, dob = %s, gender = %s
                    WHERE userid = %s
                """
                cursor.execute(sql, (password, name, nickname, dob, gender, user_id))
            else:
                sql = """
                    UPDATE member
                    SET name = %s, nickname = %s, dob = %s, gender = %s
                    WHERE userid = %s
                """
                cursor.execute(sql, (name, nickname, dob, gender, user_id))

            return redirect('/admin/dashboard')

        # ✅ GET 요청 처리: 수정폼 보여줄 때
        cursor.execute("SELECT * FROM member WHERE userid = %s", (user_id,))
        member = cursor.fetchone()

    return render_template('edit_user.html', member=member)



# 목록
@app.route('/api/posts', methods=['GET'])
def list_posts():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    title,
                    author_nickname AS author,
                    content,
                    DATE_FORMAT(post_date, '%Y-%m-%d') AS date
                FROM posts
                ORDER BY post_date DESC
            """)
            rows = cur.fetchall()
        return jsonify(rows), 200
    finally:
        conn.close()

# 단건 조회(상세)
@app.route('/api/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    user_id,
                    title,
                    author_nickname AS author,
                    content,
                    DATE_FORMAT(post_date, '%%Y-%%m-%%d %%H:%%i') AS date
                FROM posts
                WHERE id = %s
            """, (post_id,))
            row = cur.fetchone()
        if not row:
            return jsonify({"error": "존재하지 않는 글입니다."}), 404
        return jsonify(row), 200
    finally:
        conn.close()

# 작성
@app.route('/api/posts', methods=['POST'])
def create_post():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.get_json() or {}
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()
    if not title or not content:
        return jsonify({"error": "제목과 내용을 입력하세요."}), 400

    conn = get_db_connection()  # autocommit=True
    try:
        with conn.cursor() as cur:
            # 닉네임 조회
            cur.execute("SELECT nickname FROM member WHERE userid = %s", (user_id,))
            me = cur.fetchone()
            if not me:
                return jsonify({"error": "사용자 정보를 찾을 수 없습니다."}), 404
            nickname = me["nickname"]

            # INSERT
            cur.execute("""
                INSERT INTO posts (user_id, author_nickname, title, content)
                VALUES (%s, %s, %s, %s)
            """, (user_id, nickname, title, content))
            new_id = cur.lastrowid

            # 방금 글 재조회 (실패해도 폴백 보장)
            try:
                cur.execute("""
                    SELECT
                        id,
                        user_id,
                        title,
                        author_nickname AS author,
                        content,
                        DATE_FORMAT(post_date, '%%Y-%%m-%%d %%H:%%i') AS date
                    FROM posts
                    WHERE id = %s
                """, (new_id,))
                created = cur.fetchone()
                if created:
                    return jsonify(created), 201
            except Exception as e:
                print("WARN: SELECT after INSERT failed:", e)

            # 폴백 응답
            from datetime import datetime
            return jsonify({
                "id": new_id,
                "user_id": user_id,
                "title": title,
                "author": nickname,
                "content": content,
                "date": datetime.now().strftime("%%Y-%%m-%%d %%H:%%M"),
            }), 201
    except Exception as e:
        print("❌ create_post error:", e)
        return jsonify({"error": "서버 오류"}), 500
    finally:
        conn.close()

# 수정
@app.route('/api/posts/<int:post_id>', methods=['PUT'])
def update_post(post_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    data = request.get_json() or {}
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()
    if not title or not content:
        return jsonify({"error": "제목과 내용을 입력하세요."}), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 본인 글인지 확인
            cur.execute("SELECT user_id FROM posts WHERE id = %s", (post_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "존재하지 않는 글입니다."}), 404
            if row['user_id'] != user_id:
                return jsonify({"error": "권한이 없습니다."}), 403

            # 수정
            cur.execute("""
                UPDATE posts
                SET title = %s, content = %s
                WHERE id = %s
            """, (title, content, post_id))

            # 수정 후 재조회
            cur.execute("""
                SELECT
                    id,
                    user_id,
                    title,
                    author_nickname AS author,
                    content,
                    DATE_FORMAT(post_date, '%%Y-%%m-%%d %%H:%%i') AS date
                FROM posts
                WHERE id = %s
            """, (post_id,))
            updated = cur.fetchone()
            return jsonify(updated), 200
    except Exception as e:
        print("❌ update_post error:", e)
        return jsonify({"error": "서버 오류"}), 500
    finally:
        conn.close()

# 삭제
@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
def delete_post(post_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "로그인이 필요합니다."}), 401

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 본인 글인지 확인
            cur.execute("SELECT user_id FROM posts WHERE id = %s", (post_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "존재하지 않는 글입니다."}), 404
            if row['user_id'] != user_id:
                return jsonify({"error": "권한이 없습니다."}), 403

            cur.execute("DELETE FROM posts WHERE id = %s", (post_id,))
            return jsonify({"message": "삭제되었습니다."}), 200
    finally:
        conn.close()

# 로그인 상태 조회
@app.route('/api/me', methods=['GET'])
def api_me():
    uid = session.get('user_id')
    if not uid:
        return jsonify({"logged_in": False}), 200

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT userid, nickname FROM member WHERE userid=%s", (uid,))
            u = cur.fetchone()
        if not u:
            return jsonify({"logged_in": False}), 200
        return jsonify({
            "logged_in": True,
            "userid": u["userid"],
            "nickname": u.get("nickname", "")
        }), 200
    finally:
        conn.close()


# ✅ ✅ 마지막에 딱 한 번만 있어야 함!
if __name__ == '__main__':
    app.run(debug=True)