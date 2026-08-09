import base64
import os
import requests
import shutil

def get_sphinx_examples():
    """
    Get the list of example notebooks that are actually rendered in the Sphinx documentation.

    Returns
    -------
    list of str
        The rendered example notebooks.
    """
    examples_rst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.rst")
    with open(examples_rst_path, "r") as file_:
        examples_rst = file_.read()

    examples = examples_rst.split("\n\n")[-1].split("\n")

    return [example.strip() for example in examples]

def download_example_notebooks(
        examples_path,
        images_path=None,
        examples_repo_owner="holodyne",
        examples_repo_name="slmsuite-examples",
    ):
    """
    Download example notebooks.

    Note
    ~~~~
    GitHub API only supports downloading files up to 100 MB.
    """
    try:
        os.makedirs(examples_path, exist_ok=True)
        if images_path:
            os.makedirs(images_path, exist_ok=True)

        # First check if the examples are in a nearby directory
        # (e.g. if the user has cloned the examples repo).
        examples_repo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "slmsuite-examples", "examples"
        )

        # If expected clone location exists, copy from there.
        if os.path.isdir(examples_repo_path):
            for file_name in os.listdir(examples_repo_path):
                if file_name[-6:] == ".ipynb" or file_name[-4:] == ".gif":
                    print("Copying", file_name, "from local examples repo")

                    file_path = os.path.join(examples_repo_path, file_name)

                    if file_name[-6:] == ".ipynb":
                        shutil.copy(file_path, examples_path)
                    elif images_path is not None:
                        image_path = os.path.join(images_path, file_name)
                        shutil.copy(file_path, image_path)
        # Otherwise, download from GitHub API.
        else:
            tree_url = (
                "https://api.github.com/repos/{}/{}/git/trees/main?recursive=1"
                "".format(examples_repo_owner, examples_repo_name)
            )
            tree_response = requests.get(tree_url).json()
            for path_object in tree_response["tree"]:
                path_str = path_object["path"]
                if path_str[0:9] == "examples/" and ((path_str[-6:] == ".ipynb") or (path_str[-4:] == ".gif")):
                    print("Downloading", path_str)
                    file_name = path_str[9:]
                    file_url = (
                        "https://api.github.com/repos/{}/{}/git/blobs/{}"
                        "".format(examples_repo_owner, examples_repo_name, path_object["sha"])
                    )
                    file_url2 = (
                        "https://github.com/{}/{}/blob/main/{}?raw=true"
                        "".format(examples_repo_owner, examples_repo_name, path_str)
                    )
                    if path_str[-6:] == ".ipynb":
                        file_path = os.path.join(examples_path, file_name)
                        file_response = requests.get(file_url).json()
                        file_content = file_response["content"]
                        file_str = base64.b64decode(file_content.encode("utf8")).decode("utf8")
                        with open(file_path, "w", encoding='utf8') as file_:
                            file_.write(file_str)
                    elif path_str[-4:] == ".gif" and images_path is not None:
                        file_path = os.path.join(examples_path, file_name)
                        with open(file_path, "wb") as file_:
                            file_.write(requests.get(file_url2).content)

                        image_path = os.path.join(images_path, file_name)
                        shutil.copy(file_path, image_path)
    except OSError as e:
        print(
            "WARNING: Not downloading example notebooks because they have already been downloaded. "
            "Update the examples by deleting the `_examples` directory. Error:\n{}".format(e)
        )
    except BaseException as e:
        print(
            "WARNING: Unable to download example notebooks. "
            "Building without examples. Error:\n{}".format(e)
        )
