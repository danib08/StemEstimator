import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

class App(tk.Tk):
    """Main application class, represents the main window.

    :param title: The title of the window.
    :type title: str
    :param size: The size of the window in pixels.
    :type size: (int, int)
    """
    def __init__(self, title, size):
        """Constructor method. Sets window properties and creates a 
        container to stack frames on top of each other.
        """
        super().__init__()

        # Main setup    
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(size[0], size[1])
        self.resizable(False, False) 
        self.iconbitmap('./assets/icon.ico')   

        # The container will stack frames on top of each other
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {} # Frame dictionary
        for F in (StartPage, SplashPage): #TODO: add additional pages
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            # Add each frame to dictionary
            self.frames[page_name] = frame

            # Position frames in the same location
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        """Shows a tkinter frame for the given page name.

        :param page_name: The name of the frame to show.
        :type page_name: str
        :return: None
        """
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):
    """The start page of the application.

    :param parent: The parent widget.
    :type parent: tk.Tk
    :param controller: The main application controller.
    :type controller: :class:`gui.App`
    """
    def __init__(self, parent, controller):
        """Constructor method. Sets the controller and creates the widgets.
        """
        super().__init__(parent)
        self.controller = controller
        self.point_cloud_manager = None
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
    
    def make_widgets(self):
        """Creates the widgets for the frame.

        :return: None
        """
        self.header_font = ("Helvetica", 18, "bold")
        self.style = ttk.Style()
        self.style.configure('s.TButton', font=('Helvetica', 10, "bold"), foreground="green")
        self.style.configure('b.TButton', font=('Helvetica', 16, "bold"), foreground="green")

        self.canvas = tk.Canvas(self, bg="red", width=800, height=600)
        self.bg_image = tk.PhotoImage(file='./assets/bg_image.png')
        self.file_button = ttk.Button(self, text="Buscar...", style='s.TButton', 
                                      command=self.browse_file )
        self.file_entry = ttk.Entry(self, state="disabled", width=40)
        self.start_button = ttk.Button(self, text="Iniciar", style='b.TButton', state="disabled")
        
    def place_widgets(self):
        """Places the canvas and the widgets.

        :return: None
        """
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")
        self.canvas.create_rectangle(200, 100, 600, 400, fill="#32612d", 
                                outline="", stipple='gray75') 
        self.canvas.create_text(400, 130, text="Estimador de Tallos", 
                                font=self.header_font, fill="white")
        self.canvas.create_text(380, 200, text="Subir un archivo de nube de puntos", 
                                font=("Helvetica", 16), fill="white")
        self.canvas.create_window(340, 230, window=self.file_entry)
        self.canvas.create_window(535, 230, window=self.file_button)
        self.canvas.create_window(400, 300, window=self.start_button)

class SplashPage(tk.Frame):
    """A splash page for the application.
    :param parent: The parent widget.
    :type parent: tk.Tk
    :param controller: The main application controller.
    :type controller: :class:`gui.App`
    """
    def __init__(self, parent, controller):
        """Constructor method. Sets the controller and creates the widgets.
        """
        super().__init__(parent)
        self.controller = controller

        label = ttk.Label(self, text="Processing...")
        label.pack(side="top", fill="x", pady=10)

if __name__ == "__main__":
    width, height = 800, 600
    app = App("Estimador de Tallos", (width, height))
    app.mainloop()