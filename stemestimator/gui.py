import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

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

        # main setup    
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(size[0], size[1])
        self.resizable(False, False)    

        # the container will stack frames on top of each other
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {} # frame dictionary
        for F in (StartPage, SplashPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            # add each frame to dictionary
            self.frames[page_name] = frame

            # position frames in the same location
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        """Shows a tkinter frame for the given page name.

        :param page_name: The name of the frame to show.
        :type page_name: str
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
        self.make_widgets()
        self.place_widgets()

    def browse_file(self):
        """Opens a file dialog to select a file and inserts the file path
        into the entry.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.insert(0, file_path)

    def make_widgets(self):
        """Creates the widgets for the frame.
        """
        header_font = ("Helvetica", 18, "bold", "italic")
        self.header = ttk.Label(self, text="Estimador de Tallos", font=header_font)
        self.file_label = ttk.Label(self, text="Subir un archivo de nube de puntos:")
        self.open_button = ttk.Button(self, text="Buscar...", command=self.browse_file)
        self.file_entry = ttk.Entry(self)
        
    def place_widgets(self):
        """Places the widgets in the frame.
        """
        self.header.pack(side="top", fill="x", pady=10)
        self.file_label.pack(side="top", pady=10)
        self.open_button.pack(side="top", pady=10)
        self.file_entry.pack(side="top", pady=10)

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
    app = App("Estimador de Tallos", (800, 600))
    app.mainloop()