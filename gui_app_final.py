import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QPixmap, QFont, QMovie
from PyQt5.QtCore import Qt, QTimer
from PIL import Image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pneumonia Detection")
        self.setGeometry(100, 100, 1000, 600)
        self.setMinimumSize(900, 500)

        # 🎞️ Background animation
        self.movie_label = QLabel(self)
        self.movie_label.setGeometry(0, 0, self.width(), self.height())
        self.movie = QMovie("background.gif")
        self.movie.setScaledSize(self.movie_label.size())
        self.movie_label.setMovie(self.movie)
        self.movie.start()

        # 🔳 Overlay to make text readable
        self.overlay = QLabel(self)
        self.overlay.setGeometry(0, 0, self.width(), self.height())
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 120);")

        # 🧾 Title
        self.label = QLabel("Upload a Chest X-ray Image", self)
        self.label.setGeometry(200, 20, 600, 40)
        self.label.setFont(QFont("Arial", 20, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; background: transparent;")

        # 📷 Image area
        self.img_label = QLabel(self)
        self.img_label.setGeometry(100, 80, 300, 300)
        self.img_label.setStyleSheet("border: 2px dashed white; background: rgba(255,255,255,40);")

        # 📤 Upload button
        self.button = QPushButton("Upload Image", self)
        self.button.setGeometry(150, 400, 200, 40)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5A9BD4;
            }
        """)
        self.button.clicked.connect(self.load_image)

        # 🔍 Prediction Label
        self.result_label = QLabel("", self)
        self.result_label.setGeometry(100, 460, 300, 40)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: white; background: transparent;")

        # 📘 Side Panel: Medical Tips or Quotes
        self.side_panel = QTextEdit(self)
        self.side_panel.setGeometry(450, 80, 500, 300)
        self.side_panel.setReadOnly(True)
        self.side_panel.setFont(QFont("Arial", 12))
        self.side_panel.setStyleSheet("background-color: rgba(255, 255, 255, 180); color: black; border-radius: 10px; padding: 10px;")

        # 🌀 Animation state
        self.full_text = ""
        self.current_text = ""
        self.char_index = 0
        self.text_timer = QTimer()
        self.text_timer.timeout.connect(self.animate_text)

        # 🔬 Load Model
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        try:
            self.model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device('cpu')))
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load model: {e}")
            sys.exit()

    def resizeEvent(self, event):
        """Auto-resize background and overlay on window maximize or resize"""
        self.movie_label.setGeometry(0, 0, self.width(), self.height())
        self.movie.setScaledSize(self.movie_label.size())
        self.overlay.setGeometry(0, 0, self.width(), self.height())

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio)
            self.img_label.setPixmap(pixmap)

            prediction = self.predict(path)
            if prediction == 1:
                self.full_text = "Prediction: Pneumonia❗"
                self.result_label.setStyleSheet("color: red; background: transparent;")
                self.side_panel.setText(
                    "🔴 **Medical Guidance:**\n\n"
                    "💊 Medication:\n• Antibiotics as prescribed\n• Fever reducers (e.g., paracetamol)\n\n"
                    "🛡️ Prevention:\n• Vaccination\n• Avoid smoking\n• Maintain Good hygiene\n\n"
                    "🩺 When to See a Doctor:\n• Breathing difficulty\n• High fever\n• Chest pain\n"
                )
            else:
                self.full_text = "Prediction: Normal ✅"
                self.result_label.setStyleSheet("color: lightgreen; background: transparent;")
                self.side_panel.setText(
                    "🌟 *Your lungs look healthy!*\n\n"
                    "“Take care of your body. It’s the only place you have to live.”\n\n"
                    "Stay active, eat well, and keep smiling! 😊"
                )

            # Reset animation
            self.char_index = 0
            self.current_text = ""
            self.result_label.setText("")
            self.text_timer.start(50)

    def animate_text(self):
        if self.char_index < len(self.full_text):
            self.current_text += self.full_text[self.char_index]
            self.result_label.setText(self.current_text)
            self.char_index += 1
        else:
            self.text_timer.stop()

    def predict(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)
            prob = torch.sigmoid(output).item()
            print(f"Model confidence: {prob:.4f}")
            return 1 if prob > 0.4 else 0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
