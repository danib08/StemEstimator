import tkinter as tk
from pages import StartPage, ProcessingPage

class App(tk.Tk):
    """Main application class, represents the main window.

    :param title: The title of the window.
    :type title: str
    :param width: The width of the window in pixels.
    :type width: int
    :param height: The height of the window in pixels.
    :type height: int
    """
    def __init__(self, title, width, height):
        """Constructor method. Sets window properties and creates a 
        container to stack frames on top of each other.
        """
        super().__init__()

        # Main setup   
        self.title(title)
        self.width = width
        self.height = height
        self.minsize(width, height)
        self.resizable(False, False) 
        self.geometry(f'{width}x{height}')
        self.iconbitmap('./assets/icon.ico')  

        # The container will stack frames on top of each other
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {} # Frame dictionary
        for F in (StartPage, ProcessingPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self, width=self.width, height=self.height)
            # Add each frame to dictionary
            self.frames[page_name] = frame

            # Position frames in the same location
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name, str_message=None):
        """Shows a tkinter frame for the given page name.

        :param page_name: The name of the frame to show.
        :type page_name: str
        :param str_message: A message to pass to the new frame.
        :type str_message: str
        :return: None
        """
        frame = self.frames[page_name]
        if str_message is not None:
            frame.pass_message(str_message)
        frame.tkraise()

if __name__ == "__main__":
    width, height = 800, 600
    app = App("Estimador de Tallos", width, height)
    app.mainloop()