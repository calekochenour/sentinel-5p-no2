""" Module to work with the Box API """

import os
import glob
from boxsdk import Client, OAuth2


def create_session(config_file):
    """Authenticates to the Box API and creates a client session.

    Docs:
        https://developer.box.com/

    Parameters
    ----------
    config_file : str
        Path to the configuration file containing the Box API
        client id, client secret, and access token.

    Returns
    -------
    client : boxsdk.client.client.Client object
        Box API client session associated with authentication credentials.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Read Box application info from text file
    if os.path.exists(config_file):
        with open(config_file, "r") as app_cfg:
            CLIENT_ID = app_cfg.readline().rstrip("\n")
            CLIENT_SECRET = app_cfg.readline().rstrip("\n")
            ACCESS_TOKEN = app_cfg.readline().rstrip("\n")

            # Authenticate to the Box API
            auth = OAuth2(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                access_token=ACCESS_TOKEN,
            )

            # Create client session
            client = Client(auth)

            # Get user
            user = client.user().get()

        print(
            (
                f"Authenticated to Box API with user {user.name} and created a"
                "client session."
            )
        )

    else:
        print("Could not find configuration file.")
        client = None

    return client


def display_all_folders(client_session, root_folder=0):
    """Recursivley returns all folders within
    a specified root folder.

    Parameters
    ----------
    client_session : boxsdk.client.client.Client object
        Active Box API client session.

    root_folder : int or str, optional
        Existing folder ID within the Box API. Default value
        is 0, indicating the root folder for the user.

    Returns
    -------

    """
    # Get root folder contents
    contents = client_session.folder(folder_id=root_folder).get_items(
        limit=1000, offset=0
    )

    # Filter root folder contents to items of type 'folder'
    folders = filter(lambda x: x.type == "folder", contents)

    # Loop through each folder at root level
    for folder in folders:

        # Print folder name/id for all folders
        print(f"Folder: {folder.name}, ID: {folder.id}")

        # Run function again with new root folder defined
        display_all_folders(
            client_session=client_session, root_folder=folder.id
        )


def display_specific_folder(client_session, root_folder, target_folder_name):
    """Returns folder(s) matching a specific name.

    Parameters
    ----------
    client_session : boxsdk.client.client.Client object
        Active Box API client session.

    root_folder : int or str
        Existing folder ID within the Box API.

    targer_folder : str
        Name of the folder to find.

    Returns
    -------

    """
    # Get root folder contents
    contents = client_session.folder(folder_id=root_folder).get_items(
        limit=1000, offset=0
    )

    # Filter root folder contents to items of type 'folder'
    folders = filter(lambda x: x.type == "folder", contents)

    # Loop through each folder at root level
    for folder in folders:

        # Print folder name/id if folder name matches target folder
        # Also print parent folder names, in case of multiple matching names
        if folder.name == target_folder_name:
            box_folder = client_session.folder(folder_id=folder.id).get()
            box_folder_parent = client_session.folder(
                folder_id=box_folder.parent.id
            ).get()

            # Display folder hierarchy up to three levels
            try:
                print(
                    (
                        f"Folder: {box_folder_parent.parent.name}\\"
                        f"{box_folder.parent.name}\\{folder.name}, "
                        "ID: {folder.id}"
                    )
                )

            except AttributeError:
                print(
                    (
                        f"Folder: {box_folder.parent.name}\\{folder.name}, "
                        "ID: {folder.id}"
                    )
                )

        # Run function again with new root folder defined
        display_specific_folder(
            client_session=client_session,
            root_folder=folder.id,
            target_folder_name=target_folder_name,
        )


def display_folder_attributes(folder):
    """Show all folder attributes."""
    for attribute in folder:
        print(attribute)


def upload_files_to_box(box_folder, local_folder, file_extension=None):
    """Uploads all files with a specific file extension to the Box API.

    Parameters
    ----------
    box_folder : boxsdk.object.folder.Folder object
        Destination folder (in the Box API) for the files.

    local_folder : str
        Path to the source folder containing the files.

    file_extension : str, optional
        File extention (without period) to filter the upload.
        E.g. 'tif' to only upload GeoTiff files within the
        folder. Default value is None, indicating all files
        in the local folder will be uploded.

    Returns
    -------

    Example
    -------

    """
    # Get files into list
    if file_extension:
        files = glob.glob(os.path.join(local_folder, f"*.{file_extension}"))
    else:
        files = glob.glob(os.path.join(local_folder, "*"))

    # Upload each file to Box
    for file in files:
        try:
            box_folder.upload(file_path=file)
        except Exception as error:
            print(f"Failed to upload {os.path.basename(file)}")
            print(error)
        else:
            print(f"Uploaded {os.path.basename(file)} to {box_folder}")
