from datetime import datetime

def time_stamp():
    """Get a time stamp string.
    """
    return datetime.now().strftime("%d-%m-%H-%M")
