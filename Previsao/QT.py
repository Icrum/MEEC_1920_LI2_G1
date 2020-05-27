from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.initUI()

    def button_clicked(self):
        self.user.setText("G???")
        self.comando.setText("Comando ???")
        #self.update()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("LI Grupo 1")

        self.label = QtWidgets.QLabel(self)
        self.label.setText("Utilizador: ")
        self.label.move(150,550)
        self.user = QtWidgets.QLabel(self)
        self.user.setText(" ")
        self.user.move(250, 550)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setText("Comando: ")
        self.label_2.move(400, 550)
        self.comando = QtWidgets.QLabel(self)
        self.comando.setText(" ")
        self.comando.move(500, 550)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Iniciar")
        self.b1.clicked.connect(self.button_clicked)
        self.b1.move(10, 550)

    def update(self):
        self.label.adjustSize()


def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()
