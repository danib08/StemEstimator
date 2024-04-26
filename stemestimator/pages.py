import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

class BasePage(tk.Frame):
    """Base class for all pages in the application.
    
    :param parent: The parent widget.
    :type parent: tk.Tk
    :param controller: The main application controller.
    :type controller: :class:`gui.App`
    :param width: The width of the page in pixels.
    :type width: int
    :param height: The height of the page in pixels.
    :type height: int
    """
    def __init__(self, parent, controller, width, height):
        """Constructor method.
        """
        super().__init__(parent)
        self.controller = controller
        self.message = None

        self.width = width
        self.height = height
        self.header_font = ("Helvetica", 18, "bold")
        self.bg_image = tk.PhotoImage(file='./assets/bg_image.png')
        self.canvas = tk.Canvas(self, width=self.width, height=self.height)

        self.style = ttk.Style()
        self.style.configure('s.TButton', font=('Helvetica', 10, "bold"), foreground="green")
        self.style.configure('b.TButton', font=('Helvetica', 16, "bold"), foreground="green")
        self.set_background()

    def set_background(self):
        """Sets the background image for the page.

        :return: None
        """
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

    def pass_message(self, message):
        """Passes a message to the page.

        :param message: The message to pass.
        :type message: str
        :return: None
        """
        self.message = message

class StartPage(BasePage):
    """The start page of the application.

    :param parent: The parent widget.
    :type parent: tk.Tk
    :param controller: The main application controller.
    :type controller: :class:`gui.App`
    :param width: The width of the page in pixels.
    :type width: int
    :param height: The height of the page in pixels.
    :type height: int
    """
    def __init__(self, parent, controller, width, height):
        """Constructor method. Sets the controller and creates the widgets.
        """
        super().__init__(parent, controller, width, height)
        self.make_widgets()
        self.place_widgets()

    def browse_file(self):
        """Opens a file dialog to select a file and inserts the file path
        into the entry.

        :return: None
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            # Clear and set entry
            self.file_entry.state(['!disabled'])
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, file_path)
            self.file_entry.state(['disabled'])

            if self.file_checker(file_path):
                self.start_button.state(['!disabled'])
            else:
                messagebox.showerror("Alerta", "El archivo seleccionado no es un archivo XYZ o PCD válido.")
                self.start_button.state(['disabled'])

    def file_checker(self, file_path):
        """Checks if file is of .xyz or .pcd type.

        :param file_path: The path to the input file.
        :type file_path: str
        :return: True if the file is of .xyz or .pcd type, False otherwise.
        :rtype: bool
        """
        if not (file_path.endswith('.xyz') or file_path.endswith('.pcd')):
            return False
        
        if file_path.endswith('.xyz'):
            # XYZ format check (minimum 3 numeric columns)
            try:
                with open(file_path, 'r') as file:
                    for _ in range(100):
                        line = file.readline().strip()
                        columns = line.split()
                        if len(columns) >= 3:
                            try:
                                for value in columns:
                                    float(value)
                            except ValueError:
                                return False 
                        else:
                            return False
                    return True
            except FileNotFoundError:
                return False
        else:
            return True
        
    def go_to_processing(self):
        """Shows the processing page.

        :return: None
        """
        file_path = self.file_entry.get()
        self.controller.start_processing(file_path)
    
    def make_widgets(self):
        """Creates the widgets for the frame.

        :return: None
        """
        self.file_button = ttk.Button(self, text="Buscar...", style='s.TButton', 
                                      command=self.browse_file )
        self.file_entry = ttk.Entry(self, state="disabled", width=40)
        self.start_button = ttk.Button(self, text="Iniciar", style='b.TButton', state="disabled",
                                       command=self.go_to_processing)
        
    def place_widgets(self):
        """Places the widgets on the frame.

        :return: None
        """
        self.canvas.create_rectangle(200, 100, 600, 400, fill="#32612d", 
                                outline="", stipple='gray75') 
        self.canvas.create_text(400, 130, text="Estimador de Tallos", 
                                font=self.header_font, fill="white")
        self.canvas.create_text(380, 200, text="Subir un archivo de nube de puntos", 
                                font=("Helvetica", 16), fill="white")
        self.canvas.create_window(340, 230, window=self.file_entry)
        self.canvas.create_window(535, 230, window=self.file_button)
        self.canvas.create_window(400, 300, window=self.start_button)

class ProcessingPage(BasePage):
    """The processing page for the application.
    :param parent: The parent widget.
    :type parent: tk.Tk
    :param controller: The main application controller.
    :type controller: :class:`gui.App`
    :param width: The width of the page in pixels.
    :type width: int
    :param height: The height of the page in pixels.
    :type height: int
    """
    def __init__(self, parent, controller, width, height):
        """Constructor method. Sets the controller and creates the widgets.
        """
        super().__init__(parent, controller, width, height)
        self.place_widgets()

    def place_widgets(self):
        """Places the widgets on the frame.

        :return: None
        """ 
        self.canvas.create_rectangle(200, 100, 600, 400, fill="#32612d", 
                                outline="", stipple='gray75') 
        self.canvas.create_text(400, 130, text="Procesando nube de puntos...", 
                                font=self.header_font, fill="white")