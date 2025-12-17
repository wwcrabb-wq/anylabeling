"""Generic worker for running tasks in background threads."""

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class GenericWorker(QObject):
    """Generic worker for running tasks in background threads.
    
    This worker wraps a function and its arguments to be executed in a thread.
    It emits a finished signal when the function completes.
    """
    
    finished = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        """Initialize the worker.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        """Execute the worker function and emit finished signal."""
        self.func(*self.args, **self.kwargs)
        self.finished.emit()
