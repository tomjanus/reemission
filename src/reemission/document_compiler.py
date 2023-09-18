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
    """ """
    def __init__(self, document: Document) -> None:
        self.document = document

    def generate_pdf(
            self,
            filepath=None,
            *,
            clean=True,
            clean_tex=True,
            compiler=None,
            compiler_args=None,
            silent=True,
            compilations: int = 2
        ):
            """Generate a pdf file from the document.

            Args
            ----
            filepath: str
                The name of the file (without .pdf), if it is `None` the
                ``default_filepath`` attribute will be used.
            clean: bool
                Whether non-pdf files created that are created during compilation
                should be removed.
            clean_tex: bool
                Also remove the generated tex file.
            compiler: `str` or `None`
                The name of the LaTeX compiler to use. If it is None, PyLaTeX will
                choose a fitting one on its own. Starting with ``latexmk`` and then
                ``pdflatex``.
            compiler_args: `list` or `None`
                Extra arguments that should be passed to the LaTeX compiler. If
                this is None it defaults to an empty list.
            silent: bool
                Whether to hide compiler output
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
                        with open("/dev/null", "w") as nullfile:
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
                        print(output.decode())

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