import os
import sys
import subprocess
import time
import random                                    # random kainos produktams
from PyQt6 import QtWidgets, QtCore, QtGui       # PyQt6 – naudojamas sukurti self-service checkout 

#---------------------------------------------
# Jei OpenCV (cv2) nera ideta programa vistiek veik  tiesiog neveiks QR skanavimas.
CAMERA_AVAILABLE = False
try:
    import cv2                                   # leidžia skanuoti QR kodą per kamerą
    CAMERA_AVAILABLE = True
except Exception as e:
    print("Camera import error:", repr(e))
    CAMERA_AVAILABLE = False

#---------------------------------------------
# Paprastas "Data base" – kokie produktai yra ribojami (age restricted)
RESTRICTED_DB = {
    "Beer": {"name": "Beer", "restricted": True},   # Pvz.:  "Beer" bus laikomas restricet item
}

def get_base_dir():                        # kaip gaunam dabartinio failo directiona
    return os.path.dirname(os.path.abspath(__file__))

def get_webcam_script_path():                # kaip gaunam app.py (amžiaus check app)
    base = get_base_dir()
    return os.path.join(base, "app.py")

def get_cigarettes_image_path():              # kaip gaunas cigaretes (cigarettes.png)
    base = get_base_dir()
    return os.path.join(base, "cigarettes.png")

def get_alcohol_image_path():                # Kelias iki alkoholio paveiksliuko (alcohol.png)
    base = get_base_dir()
    return os.path.join(base, "alcohol.png")

def lookup_barcode(code: str):              # Pagal QR teksta paziuri, ar produktas yra DataBase. # Jei ner – grazina paprasta neredtricted item ("Item scanned").
    return RESTRICTED_DB.get(code, {"name": "Item scanned", "restricted": False})

#---------------------------------------------
# Worker: Age Check Subprocess
class AgeCheckWorker(QtCore.QObject): # Paleidžia app.py kaip atskirą procesą (subprocess),laukia kol tas procesas išmes eilutes su: - 'LOCKED AGE: ...' - 'RESULT: ...'    ir perduoda rezultatą atgal GENEREL USER INTERACE per finished signalą.

    finished = QtCore.pyqtSignal(str, object, bool, str)

    def __init__(self, barcode: str):
        super().__init__()
        self.barcode = barcode           # šiuo metu nenaudojam app.py viduje, bet paliekam ateičiai (logavimui ir pan.)

    @QtCore.pyqtSlot()
    def run(self):
        script = get_webcam_script_path()

        # Windows'e paslepia subprocess langą (Mac/Linux – tiesiog ignoruojama)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            p = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                creationflags=creationflags,
            )
        except Exception as e:
            self.finished.emit("", None, False, f"Failed to start age-check app: {e}")
            return

        decision = None
        locked_age = None

        try:
            while True:
                if p.stdout is None:
                    break

                line = p.stdout.readline()
                if line == "" and p.poll() is not None:       # Procesas baigesi ir daugiau eiluciu nebera
                    break

                if not line:                  # Tiesiog nieko neatėjo – truputį palaukiam ir skaitom toliau
                    time.sleep(0.1)
                    continue

                line = line.strip()

                if "LOCKED AGE:" in line:                     # Pvz.: "LOCKED AGE: 23.7"
                    try:
                        locked_age = float(line.split(":", 1)[1].strip())
                    except Exception:
                        locked_age = None

                if "RESULT:" in line:                         # Pvz.: "RESULT: No ID required"
                    decision = line.split(":", 1)[1].strip()
                    break
        finally:
            try:
                p.terminate()
            except Exception:
                pass

        if decision:
            self.finished.emit(decision, locked_age, True, "")
        else:
            self.finished.emit("", locked_age, False, "No decision returned from age-check app.")

#---------------------------------------------
# Kamera Scanneris
class CameraScannerWorker(QtCore.QObject):        # Skaito QR kodą iš kameros naudojant OpenCV QRCodeDetector. Nerodo jokio OpenCV lango, dirba fone thread'e.
    barcode_found = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)              # info arba error message (empty jei OK)

    @QtCore.pyqtSlot()
    def run(self):
        if not CAMERA_AVAILABLE:
            self.finished.emit("Camera (OpenCV) not available.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.finished.emit("Failed to open camera.")
            return

        detector = cv2.QRCodeDetector()
        found_code = None
        start_time = time.time()
        timeout_seconds = 15                       # jei per 15 s neranda QR – baigia

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            data, bbox, _ = detector.detectAndDecode(frame)
            if data:
                found_code = data.strip()
                break

            time.sleep(0.02)
            if time.time() - start_time > timeout_seconds:
                break

        cap.release()

        if found_code:
            self.barcode_found.emit(found_code)        # perduoda rasta QR teksta i graphical user interface
            self.finished.emit("")
        else:
            self.finished.emit("No QR code detected (timed out).")

#---------------------------------------------
# MAIN WINDOW - Graphical user interface
class SelfCheckoutWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.webcam_proc = None
        self.current_restricted_barcode: str | None = None
        self.current_restricted_name: str | None = None
        self.total_amount = 0.0                             # bendra basket suma

    
        self.restricted_in_basket = False             # ar yra bent vienas restricted baskete
        self.age_check_for_payment = False            # ar dabar amziaus check paleistas edl Pay mygtuko
        self.age_verified_override = False            # ar admin/AI jau patvirtino age pirkimui
        self.current_restricted_in_basket = False     # ar dabartinis restricted item jau idetas i basket

        self.setWindowTitle("Self-Service Checkout")
        self.setMinimumSize(1100, 650)

        self._build_ui()
        self._apply_styles()
        self._setup_shortcuts()

    #-----------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        outer_layout = QtWidgets.QVBoxLayout(central)
        outer_layout.setSpacing(16)
        outer_layout.setContentsMargins(24, 16, 24, 24)

        # ------- viršus: pavadinimas + GIF + admin -------
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(12)

        self.title_label = QtWidgets.QLabel("Self-Service Checkout")
        self.title_label.setObjectName("titleLabel")
        top_bar.addWidget(self.title_label, stretch=1)
        top_bar.addStretch()

        # Animated GIF logo (petka.gif)
        logo_label = QtWidgets.QLabel()
        gif_path = os.path.join(get_base_dir(), "petka.gif")

        if os.path.exists(gif_path):
            movie = QtGui.QMovie(gif_path)
            movie.setScaledSize(QtCore.QSize(150, 150))
            logo_label.setMovie(movie)
            movie.start()

        logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        top_bar.addWidget(logo_label)

        self.admin_button = QtWidgets.QPushButton("Admin")
        self.admin_button.setObjectName("adminButton")
        self.admin_button.setToolTip("Admin login (or press Ctrl+Shift+A)")
        self.admin_button.clicked.connect(self.show_admin_login)
        top_bar.addWidget(self.admin_button, stretch=0)

        outer_layout.addLayout(top_bar)

        # main screen: LEFT / RIGHT 
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(24)

        # LEFT: ekranas + mygtukai + statusas
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(18)

        self.screen = QtWidgets.QFrame()
        self.screen.setObjectName("screenFrame")
        s_layout = QtWidgets.QVBoxLayout(self.screen)
        s_layout.setContentsMargins(20, 20, 20, 20)
        s_layout.setSpacing(10)

        self.screen_main_label = QtWidgets.QLabel("Scan your items")
        self.screen_main_label.setObjectName("screenMain")
        self.screen_main_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.screen_hint_label = QtWidgets.QLabel(
            "To scan a product, press the green "
            "<b><span style='color:#3CCB5A;'>Scan via Camera</span></b> button.<br><br>"
            "Hold the QR code steadily in front of the camera — it will be used as the item barcode.<br><br>"
            "If you are only purchasing alcohol or cigarettes, press the red "
            "<b><span style='color:#FF4C4C;'>Age-Restricted Items</span></b> button."
        )
        self.screen_hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.screen_hint_label.setWordWrap(True)

        s_layout.addWidget(self.screen_main_label)
        s_layout.addWidget(self.screen_hint_label)
        s_layout.addStretch()

        left.addWidget(self.screen, stretch=2)

        # Mygtukai: restricted / stop webcam / scan
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(16)

        self.btn_cigarettes = QtWidgets.QPushButton("Age-Restricted Items")
        self.btn_cigarettes.setObjectName("cigButton")
        self.btn_cigarettes.clicked.connect(self.show_restricted_items_dialog)

        self.btn_stop = QtWidgets.QPushButton("Stop Webcam")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_webcam)

        self.btn_scan = QtWidgets.QPushButton("Scan via Camera")
        self.btn_scan.setObjectName("scanButton")
        self.btn_scan.clicked.connect(self.start_camera_scanner)

        btn_row.addWidget(self.btn_cigarettes)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_scan)

        left.addLayout(btn_row)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(16)

        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        bottom_row.addWidget(self.status_label)
        left.addLayout(bottom_row)

        # RIGHT: krepšelis + Pay + Age info
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        basket_title = QtWidgets.QLabel("Basket")
        basket_title.setObjectName("basketTitle")
        right.addWidget(basket_title)

        self.receipt_list = QtWidgets.QListWidget()
        self.receipt_list.setObjectName("receiptList")
        right.addWidget(self.receipt_list, stretch=2)

        self.total_label = QtWidgets.QLabel("Total: €0.00")
        self.total_label.setObjectName("totalLabel")
        right.addWidget(self.total_label)

        # Pay
        right.addSpacing(8)
        self.pay_button = QtWidgets.QPushButton("Pay")
        self.pay_button.clicked.connect(self.show_payment_options)
        right.addWidget(self.pay_button)
        right.addSpacing(12)

        age_title = QtWidgets.QLabel("Age Verification")
        age_title.setObjectName("ageTitle")
        right.addWidget(age_title)

        self.age_result_label = QtWidgets.QLabel("No age checks performed yet.")
        self.age_result_label.setWordWrap(True)
        right.addWidget(self.age_result_label)

        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=2)
        outer_layout.addLayout(main_layout)

    #-----------------------------------------
    def _apply_styles(self):
        # (palikau tavo gražų stilių nepakeistą)
        self.setStyleSheet("""
        QMainWindow {
            background-color: #0f172a;
        }

        QLabel {
            color: #e5e7eb;
            font-size: 14px;
        }

        #titleLabel {
            font-size: 36px;
            font-weight: 800;
        }

        #screenFrame {
            background-color: #111827;
            border-radius: 18px;
            border: 1px solid #1f2937;
        }

        #screenMain {
            font-size: 20px;
            font-weight: 600;
        }

        QPushButton {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 14px 20px;
            font-size: 15px;
            color: #e5e7eb;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: #1f2937;
        }

        QPushButton#cigButton {
            background-color: #7f1d1d;
            border: 1px solid #991b1b;
        }

        QPushButton#cigButton:hover {
            background-color: #991b1b;
        }

        QPushButton#stopButton {
            background-color: #1d4ed8;
            border: 1px solid #2563eb;
        }

        QPushButton#stopButton:disabled {
            background-color: #1e293b;
            border-color: #111827;
            color: #64748b;
        }

        QPushButton#scanButton {
            background-color: #065f46;
            border: 1px solid #047857;
        }

        QPushButton#adminButton {
            padding: 8px 14px;
            font-size: 12px;
            border-radius: 999px;
            background-color: #172554;
            border: 1px solid #1d4ed8;
        }

        #receiptList {
            background-color: #020617;
            border-radius: 12px;
            border: 1px solid #1f2937;
            color: #f9fafb;
        }

        #basketTitle {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        #statusLabel {
            color: #9ca3af;
            font-size: 12px;
        }

        #ageTitle {
            font-size: 16px;
            font-weight: 600;
            color: #fbbf24;
        }

        #totalLabel {
            color: #f9fafb;
            font-size: 14px;
        }
        """)

    def _setup_shortcuts(self):
        admin_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+A"), self)
        admin_shortcut.activated.connect(self.show_admin_login)

    #-----------------------------------------
    # Helpers / status
    def set_status(self, text: str):
        self.status_label.setText(text)

    def add_item_with_random_price(self, name: str, restricted: bool = False):
        """Prideda prekę su atsitiktine kaina ir atnaujina bendrą sumą."""
        price = random.uniform(5.0, 20.0)
        self.total_amount += price

        item_text = f"{name} - €{price:.2f}"
        item = QtWidgets.QListWidgetItem(item_text)

        if restricted:
            item.setForeground(QtGui.QBrush(QtGui.QColor("#f87171")))  # raudona restricted

        self.receipt_list.addItem(item)
        self.receipt_list.scrollToBottom()
        self.total_label.setText(f"Total: €{self.total_amount:.2f}")

    #-----------------------------------------
    # Webcam launcher (legacy – naudojama tik jei norėsi atskiros webcam app)
    def launch_webcam(self):
        script = get_webcam_script_path()
        if not os.path.exists(script):
            QtWidgets.QMessageBox.critical(self, "Error", f"Webcam script not found:\n{script}")
            return

        if self.webcam_proc and self.webcam_proc.poll() is None:
            QtWidgets.QMessageBox.information(
                self,
                "Already running",
                "Webcam app is already running."
            )
            return

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self.webcam_proc = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                creationflags=creationflags,
            )
            self.btn_cigarettes.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.set_status("Webcam app started.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Launch error", str(e))
            self.set_status("Failed to start webcam app.")

    def stop_webcam(self):
        """Sustabdytų atskirą webcam app, jei ji būtų paleista per launch_webcam."""
        if self.webcam_proc and self.webcam_proc.poll() is None:
            try:
                self.webcam_proc.terminate()
                self.webcam_proc.wait(timeout=5)
            except Exception:
                try:
                    self.webcam_proc.kill()
                except Exception:
                    pass

        self.webcam_proc = None
        self.btn_cigarettes.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.set_status("Webcam app stopped.")

    #-----------------------------------------
    # Barcode handling (kai randam QR koda)
    def handle_barcode(self, code: str):  #Jei produktas ribojamas – ji pridedam ir amge tikrinam tik kai pressinam Pay. Nerestriced– tiesiog pridedam i basket.
        self.receipt_list.addItem(f"Scanned code: {code}")
        self.receipt_list.scrollToBottom()

        info = lookup_barcode(code)

        if info.get("restricted"):
            # Restricted (Beer ir pan.)
            self.current_restricted_barcode = code
            self.current_restricted_name = info.get("name")
            self.restricted_in_basket = True
            self.current_restricted_in_basket = True
            self.age_verified_override = False

            self.add_item_with_random_price(info.get("name"), restricted=True)
            self.age_result_label.setText(
                "Restricted item in basket. Age will be checked at payment."
            )
            self.set_status(
                f"Restricted item added: {info.get('name')}. Age verification will run when you press Pay."
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Scanned",
                f"{info.get('name')} scanned (not restricted)."
            )
            self.add_item_with_random_price(info.get("name"), restricted=False)
            self.set_status(f"Scanned: {info.get('name')}")

    #-----------------------------------------
    # Decision interpretation helper
    def _interpret_decision_flags(self, decision: str):
        # Iš tekstinės išvados (pvz. 'No ID required') išsitraukiam 3 boolean: denied, no_id_required, id_required.
        decision_l = (decision or "").lower().strip()

        denied = (
            "deny" in decision_l
            or "denied" in decision_l
            or "refuse" in decision_l
            or "refused" in decision_l
        )

        no_id_required = (
            "no id required" in decision_l
            or "no id is required" in decision_l
            or "no need for id" in decision_l
        )

        id_required = (
            not no_id_required
            and (
                "id required" in decision_l
                or "id needed" in decision_l
                or "check id" in decision_l
                or "need id" in decision_l
                or "id check required" in decision_l
                or ("id" in decision_l and "check" in decision_l and "required" in decision_l)
            )
        )

        return denied, no_id_required, id_required

    #-----------------------------------------
    # Age check handling
    def start_age_check(self, barcode: str):
        self.set_status("Running age verification...")
        self.age_result_label.setText("Age verification in progress...")

        self.age_thread = QtCore.QThread()
        self.age_worker = AgeCheckWorker(barcode)
        self.age_worker.moveToThread(self.age_thread)

        self.age_thread.started.connect(self.age_worker.run)
        self.age_worker.finished.connect(self.on_age_check_finished)
        self.age_worker.finished.connect(self.age_thread.quit)
        self.age_worker.finished.connect(self.age_worker.deleteLater)
        self.age_thread.finished.connect(self.age_thread.deleteLater)

        self.age_thread.start()

    @QtCore.pyqtSlot(str, object, bool, str)
    def on_age_check_finished(self, decision, locked_age, success, err):
        if success:
            msg = f"Decision: {decision}"
            if locked_age is not None:
                try:
                    msg += f" (estimated age: {float(locked_age):.1f})"
                except Exception:
                    pass

            QtWidgets.QMessageBox.information(self, "Age check result", msg)
            self.age_result_label.setText(msg)
            self.set_status("Age verification completed.")

            denied, no_id_required, id_required = self._interpret_decision_flags(decision)

            if self.age_check_for_payment:
                self.age_check_for_payment = False

                if denied:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Payment blocked",
                        "Age verification failed. Ask for assistance."
                    )
                    self.set_status("Payment blocked due to failed age verification.")
                    return

                if id_required:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "ID required",
                        "The system requires ID verification by staff.\n"
                        "Payment is blocked until an admin approves the age."
                    )
                    self.age_verified_override = False
                    self.set_status("ID check required by staff. Waiting for admin override.")
                    return

                # čia patenkam, jei 'no id required' arba aiškiai teigiamas rezultatas
                self.restricted_in_basket = False
                self.age_verified_override = True
                self._open_payment_dialog()

        else:
            QtWidgets.QMessageBox.warning(self, "Age check", err or "No decision returned.")
            self.age_result_label.setText(err or "No decision returned.")
            self.set_status("Age verification failed or no decision.")

    #-----------------------------------------
    # Camera Scanner handling (QR)
    def start_camera_scanner(self):
        self.set_status("Starting camera QR scanner...")

        self.camera_thread = QtCore.QThread()
        self.camera_worker = CameraScannerWorker()
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.barcode_found.connect(self.handle_barcode)
        self.camera_worker.finished.connect(self.on_camera_scanner_finished)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)

        self.camera_thread.start()

    @QtCore.pyqtSlot(str)
    def on_camera_scanner_finished(self, message: str):
        if message:
            QtWidgets.QMessageBox.information(self, "Camera scanner", message)
            self.set_status(message)
        else:
            self.set_status("Camera scanner finished.")

    #-----------------------------------------
    # Age-Restricted Items dialog with clickable images
    def show_restricted_items_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Restricted Item")
        dialog.setModal(True)
        dialog.setMinimumSize(520, 340)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(16)

        label = QtWidgets.QLabel("Select item:")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(24)

        target_w, target_h = 220, 220  # same size for both icons

        # Cigarettes button
        cig_button = QtWidgets.QToolButton()
        cig_button.setText("Cigarettes")
        cig_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        cig_pix_path = get_cigarettes_image_path()
        cig_pix = QtGui.QPixmap(cig_pix_path) if os.path.exists(cig_pix_path) else QtGui.QPixmap()
        if not cig_pix.isNull():
            cig_pix = cig_pix.scaled(
                target_w,
                target_h,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            cig_button.setIcon(QtGui.QIcon(cig_pix))
            cig_button.setIconSize(QtCore.QSize(target_w, target_h))

        cig_button.setFixedSize(target_w + 40, target_h + 70)

        # Alcohol button
        alc_button = QtWidgets.QToolButton()
        alc_button.setText("Alcohol")
        alc_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        alc_pix_path = get_alcohol_image_path()
        alc_pix = QtGui.QPixmap(alc_pix_path) if os.path.exists(alc_pix_path) else QtGui.QPixmap()
        if not alc_pix.isNull():
            alc_pix = alc_pix.scaled(
                target_w,
                target_h,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            alc_button.setIcon(QtGui.QIcon(alc_pix))
            alc_button.setIconSize(QtCore.QSize(target_w, target_h))

        alc_button.setFixedSize(target_w + 40, target_h + 70)

        btn_row.addWidget(cig_button)
        btn_row.addWidget(alc_button)
        layout.addLayout(btn_row)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(button_box)

        # Pasirinkus restricted item – iškart pridedam į krepšelį, o amžių tikrinam tik spaudžiant Pay
        def choose_cig():
            self.current_restricted_barcode = "CIGARETTES"
            self.current_restricted_name = "Cigarettes"
            self.restricted_in_basket = True
            self.current_restricted_in_basket = True
            self.age_verified_override = False
            self.add_item_with_random_price(self.current_restricted_name, restricted=True)
            self.age_result_label.setText(
                "Restricted item in basket. Age will be checked at payment."
            )
            self.set_status("Restricted item added. Age check will run at payment.")
            dialog.accept()

        def choose_alc():
            self.current_restricted_barcode = "ALCOHOL"
            self.current_restricted_name = "Alcohol"
            self.restricted_in_basket = True
            self.current_restricted_in_basket = True
            self.age_verified_override = False
            self.add_item_with_random_price(self.current_restricted_name, restricted=True)
            self.age_result_label.setText(
                "Restricted item in basket. Age will be checked at payment."
            )
            self.set_status("Restricted item added. Age check will run at payment.")
            dialog.accept()

        cig_button.clicked.connect(choose_cig)
        alc_button.clicked.connect(choose_alc)
        button_box.rejected.connect(dialog.reject)

        dialog.exec()

    #-----------------------------------------
    # Pay button: pirmiausia age check (jei reikia), tada Card / Cash / Cancel
    def show_payment_options(self):
        if not self.restricted_in_basket or self.age_verified_override:
            self._open_payment_dialog()
            return

        self.set_status("Age verification required before payment.")
        self.age_result_label.setText("Age verification required before payment.")
        self.age_check_for_payment = True

        if not self.current_restricted_barcode:
            self.current_restricted_barcode = "BASKET_RESTRICTED"

        self.start_age_check(self.current_restricted_barcode)

    def _open_payment_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Payment Method")
        dialog.setModal(True)
        dialog.setMinimumSize(400, 220)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(16)

        label = QtWidgets.QLabel("Choose how you want to pay:")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)

        btn_card = QtWidgets.QPushButton("Card")
        btn_cash = QtWidgets.QPushButton("Cash")
        btn_cancel = QtWidgets.QPushButton("Cancel")

        btn_row.addWidget(btn_card)
        btn_row.addWidget(btn_cash)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        layout.addStretch()

        def complete_payment(method: str):
            QtWidgets.QMessageBox.information(
                self,
                "Payment",
                f"{method} payment selected."
            )
            self.receipt_list.clear()
            self.total_amount = 0.0
            self.total_label.setText("Total: €0.00")
            self.age_result_label.setText("No age checks performed yet.")
            self.set_status("Payment completed.")

            self.restricted_in_basket = False
            self.age_verified_override = False
            self.current_restricted_barcode = None
            self.current_restricted_name = None
            self.current_restricted_in_basket = False

            dialog.accept()

        btn_card.clicked.connect(lambda: complete_payment("Card"))
        btn_cash.clicked.connect(lambda: complete_payment("Cash"))
        btn_cancel.clicked.connect(dialog.reject)

        dialog.exec()

    #-----------------------------------------
    # Admin / override
    def show_admin_login(self):
        pw, ok = QtWidgets.QInputDialog.getText(
            self,
            "Admin Login",
            "Enter admin password:",
            QtWidgets.QLineEdit.EchoMode.Password
        )
        if not ok:
            return
        if pw == "1234":
            self.open_admin_menu()
        else:
            QtWidgets.QMessageBox.warning(self, "Access denied", "Incorrect password.")

    def open_admin_menu(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Admin Panel")
        dialog.setModal(True)
        dialog.setMinimumSize(400, 260)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Admin Controls")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        current_restricted = (
            self.current_restricted_name if self.current_restricted_name else "None"
        )

        info_label = QtWidgets.QLabel(
            f"System status: {self.status_label.text()}\n"
            f"Current restricted item: {current_restricted}\n"
            f"Age override: {'ON' if self.age_verified_override else 'OFF'}"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        btn_approve = QtWidgets.QPushButton("Approve Restricted Item / Age Override")
        btn_approve.clicked.connect(self.approve_restricted_item)
        layout.addWidget(btn_approve)

        layout.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        dialog.exec()

    def approve_restricted_item(self):
        if not self.restricted_in_basket and not self.current_restricted_name:
            QtWidgets.QMessageBox.information(
                self,
                "No restricted item",
                "There is no restricted item pending approval."
            )
            return

        if self.current_restricted_name and not self.current_restricted_in_basket:
            self.add_item_with_random_price(self.current_restricted_name, restricted=True)
            self.restricted_in_basket = True
            self.current_restricted_in_basket = True

        QtWidgets.QMessageBox.information(
            self,
            "Override",
            "Age overridden by admin. No further age checks for this transaction."
        )

        self.age_verified_override = True
        self.age_result_label.setText(
            "Manually approved by admin.\n"
            "No further age checks for this transaction."
        )
        self.set_status("Admin approved age (override active).")

    #-----------------------------------------
    # Window close
    def closeEvent(self, e: QtGui.QCloseEvent):
        self.stop_webcam()
        e.accept()

#---------------------------------------------
# Entrypoint – paleidžia PyQt6 aplikaciją
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SelfCheckoutWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
