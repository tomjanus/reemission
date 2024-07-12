"""
This module provides functionality for compiling LaTeX documents into PDF files.

Classes:
    BatchCompiler: A class to compile LaTeX documents into PDF files using various compilers.

Example:
    
.. code-block:: Python

    from pylatex import Document
    from your_module_name import BatchCompiler

    doc = Document()
    # Add content to your document here...
    # ...
    compiler = BatchCompiler(doc)
    compiler.generate_pdf(filepath="output", clean=True, compiler="pdflatex")
"""
import sys
import os
import subprocess
import errno
import logging
from pylatex import Document
from pylatex.utils import rm_temp_dir
from pylatex.errors import CompilerError

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class BatchCompiler:
    """
    A class to compile LaTeX documents into PDF files using various compilers.

    Attributes:
        document (Document): An instance of pylatex.Document representing the LaTeX document to be compiled.
    """
    def __init__(self, document: Document) -> None:
        """
        Initializes the BatchCompiler with a LaTeX document.

        Args:
            document (Document): An instance of pylatex.Document.
        """
        self.document = document

    def generate_pdf(self, filepath=None, *, clean=True, clean_tex=True, compiler=None,
                     compiler_args=None, silent=True, compilations: int = 2) -> None:
        """
        Generates a PDF file from the LaTeX document.

        Args:
            filepath (str, optional): The name of the file (without .pdf). If None, the 
                ``default_filepath`` attribute will be used.
            clean (bool, optional): Whether to remove non-PDF files created during compilation. 
                Defaults to True.
            clean_tex (bool, optional): Whether to remove the generated .tex file. Defaults to True.
            compiler (str or None, optional): The name of the LaTeX compiler to use. 
                If None, PyLaTeX will choose a suitable one. Defaults to None.
            compiler_args (list or None, optional): Extra arguments to pass to the LaTeX compiler. 
                Defaults to an empty list if None.
            silent (bool, optional): Whether to hide compiler output. Defaults to True.
            compilations (int, optional): The number of times to run the compiler. Defaults to 2.

        Raises:
            CompilerError: If no LaTeX compiler is found or an error occurs during compilation.
        """

        if compiler_args is None:
            compiler_args = []

        # In case of newer python with the use of the cwd parameter
        # one can avoid to physically change the directory
        # to the destination folder
        python_cwd_available = sys.version_info >= (3, 6)

        filepath = self.document._select_filepath(filepath)
        if not os.path.basename(filepath):
            filepath = os.path.join(os.path.abspath(filepath), "default_basename")
        else:
            filepath = os.path.abspath(filepath)

        cur_dir = os.getcwd()
        dest_dir = os.path.dirname(filepath)

        if not python_cwd_available:
            os.chdir(dest_dir)

        self.document.generate_tex(filepath)
        log.info("Created a LaTeX document with outputs.")

        if compiler is not None:
            compilers = ((compiler, []),)
        else:
            latexmk_args = ["--pdf"]

            compilers = (("latexmk", latexmk_args), ("pdflatex", []))

        main_arguments = ["--interaction=nonstopmode", filepath + ".tex"]

        check_output_kwargs = {}
        if python_cwd_available:
            check_output_kwargs = {"cwd": dest_dir}

        os_error = None

        for compiler, arguments in compilers:
            command = [compiler] + arguments + compiler_args + main_arguments

            try:
                for _ in range(0,compilations):
                    # Redirect output from latex compiler to a null stream
                    with open("/dev/null", "w", encoding="utf-8") as nullfile:
                        process = subprocess.Popen(
                            command, 
                            stdout=nullfile,
                            stderr=subprocess.STDOUT, 
                            **check_output_kwargs)
                        output = process.wait()
            except (OSError, IOError) as e:
                # Use FileNotFoundError when python 2 is dropped
                os_error = e

                if os_error.errno == errno.ENOENT:
                    # If compiler does not exist, try next in the list
                    continue
                raise
            except subprocess.CalledProcessError as e:
                # For all other errors print the output and raise the error
                print(e.output.decode())
                raise
            else:
                if not silent:
                    print(e.output.decode())

            if clean:
                try:
                    # Try latexmk cleaning first
                    subprocess.check_output(
                        ["latexmk", "-c", filepath],
                        stderr=subprocess.STDOUT,
                        **check_output_kwargs
                    )
                except (OSError, IOError, subprocess.CalledProcessError):
                    # Otherwise just remove some file extensions.
                    extensions = ["aux", "log", "out", "fls", "fdb_latexmk"]

                    for ext in extensions:
                        try:
                            os.remove(filepath + "." + ext)
                        except (OSError, IOError) as e:
                            # Use FileNotFoundError when python 2 is dropped
                            if e.errno != errno.ENOENT:
                                raise
                rm_temp_dir()

            if clean_tex:
                os.remove(filepath + ".tex")  # Remove generated tex file

            # Compilation has finished, so no further compilers have to be
            # tried
            break

        else:
            # Notify user that none of the compilers worked.
            raise (
                CompilerError(
                    "No LaTex compiler was found\n"
                    "Either specify a LaTex compiler "
                    "or make sure you have latexmk or pdfLaTex installed."
                )
            )

        if not python_cwd_available:
            os.chdir(cur_dir)
