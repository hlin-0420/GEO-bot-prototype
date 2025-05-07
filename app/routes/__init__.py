print("[DEBUG] routes/__init__.py: starting blueprint imports")

try:
    from .api import api_blueprint
    print("[DEBUG] routes/__init__.py: loaded api_blueprint")
except Exception as e:
    print(f"[ERROR] Failed to import api_blueprint: {e}")
    raise

try:
    from .ui import ui_blueprint
    print("[DEBUG] routes/__init__.py: loaded ui_blueprint")
except Exception as e:
    print(f"[ERROR] Failed to import ui_blueprint: {e}")
    raise

try:
    from .feedback_handler import feedback_routes
    print("[DEBUG] routes/__init__.py: loaded feedback_routes")
except Exception as e:
    print(f"[ERROR] Failed to import feedback_routes: {e}")
    raise

try:
    from .session_manager_extended import session_routes
    print("[DEBUG] routes/__init__.py: loaded session_routes")
except Exception as e:
    print(f"[ERROR] Failed to import session_routes: {e}")
    raise

try:
    from .timing_logger import timing_routes
    print("[DEBUG] routes/__init__.py: loaded timing_routes")
except Exception as e:
    print(f"[ERROR] Failed to import timing_routes: {e}")
    raise

try:
    from .upload_handler import upload_routes
    print("[DEBUG] routes/__init__.py: loaded upload_routes")
except Exception as e:
    print(f"[ERROR] Failed to import upload_routes: {e}")
    raise


def register_blueprints(app):
    print("[DEBUG] register_blueprints() called")
    app.register_blueprint(api_blueprint)
    app.register_blueprint(ui_blueprint)
    app.register_blueprint(feedback_routes)
    app.register_blueprint(session_routes)
    app.register_blueprint(timing_routes)
    app.register_blueprint(upload_routes)
    print("[DEBUG] All blueprints registered successfully")