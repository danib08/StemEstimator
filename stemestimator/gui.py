import threading
import tkinter as tk
from manager import PointCloudManager
from pages import StartPage, ProcessingPage, ResultsPage

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
        self.manager = None
        self.frames = {} # Frame dictionary

        # Main setup   
        self.title(title)
        self.width = width
        self.height = height
        self.minsize(width, height)
        self.resizable(False, False) 
        self.geometry(f'{width}x{height}')
        self.iconbitmap('./assets/icon.ico')  

        self.create_frame_dict()
        self.show_frame("StartPage")

    def create_frame_dict(self):
        """Creates a dictionary of frames to stack on top of each other.

        :return: None
        """
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        for F in (StartPage, ProcessingPage, ResultsPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self, width=self.width, height=self.height)
            # Add each frame to dictionary
            self.frames[page_name] = frame

            # Position frames in the same location
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, page_name):
        """Shows a tkinter frame for the given page name.

        :param page_name: The name of the frame to show.
        :type page_name: str
        :return: None
        """
        frame = self.frames[page_name]
        if page_name == "ResultsPage":
            frame.create()
        frame.tkraise()

    def start_processing(self, file_path):
        """Starts the processing thread for the point cloud.

        :param file_path: The path to the point cloud file.
        :type file_path: str
        :return: None
        """
        processing_thread = threading.Thread(target=self.point_cloud_processing, args=(file_path,))
        processing_thread.start()

    def point_cloud_processing(self, file_path):
        """Processes the point cloud file.

        :param file_path: The path to the point cloud file.
        :type file_path: str
        :return: None
        """
        self.show_frame("ProcessingPage")
        self.manager = PointCloudManager(file_path)
        self.manager.remove_ground()
        self.manager.normal_filtering()
        self.manager.clustering()
        self.manager.group_stems()
        self.manager.fit_ellipses()
        self.show_frame("ResultsPage")

    def show_final_point_cloud(self):
        """Shows the results of the point cloud processing.

        :return: None
        """
        self.manager.show_final_point_cloud()

    def get_tree_count(self):
        """Returns the number of trees in the point cloud.

        :return: int
        """
        return self.manager.get_tree_count()

if __name__ == "__main__":
    width, height = 800, 600
    app = App("Estimador de Tallos", width, height)
    app.mainloop()