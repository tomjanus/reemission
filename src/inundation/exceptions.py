""" Module containing exceptions for inundation calculations """


class WrongExtensionException(Exception):
    """ Custom exception raised if file has wrong extension
        extension : str
        message : str
    """
    def __init__(self, extension, message="Wrong file extension."):
        self.extension = extension
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.extension} -> {self.message}'


class NotRasterException(WrongExtensionException):
    """ Exception handler when file not raster """
    def __init__(self, extension, message="File extension not raster."):
        super().__init__(extension, message)


class NotVectorException(WrongExtensionException):
    """ Exception handler when file not vector """
    def __init__(self, extension, message="File extension not vector."):
        super().__init__(extension, message)
