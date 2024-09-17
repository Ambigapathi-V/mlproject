import sys
from src.logger import logging  # Ensure this imports the correct logger

def error_message_detail(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Properly formatting the error message with details
    error_message = f"Error occurred in python script: [{0}], line number: [{1}], error message: [{str(2)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        # Initialize the Exception class and capture the detailed error message
        self.error_message = error_message_detail(error, error_details=error_detail)
        # Log the error message in the same log file
        logging.error(self.error_message)
        super().__init__(self.error_message)
    
    def __str__(self):
        return self.error_message

