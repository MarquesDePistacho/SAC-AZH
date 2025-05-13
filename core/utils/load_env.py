import logging
import os
import zipfile


def load_env(env_path: str, tmp_dir: str="tmp") -> str:
    """
    Загружает среду Unity из указанного пути. Если путь указывает на zip-архив,
    извлекает архив во временную директорию и возвращает путь к
    извлеченному билду среды Unity.

    Args:
        env_path (str): Путь к среде (может быть zip-архивом или билдом Unity).
        tmp_dir (str):  Имя временной директории для извлечения архива.
                        По умолчанию "tmp".

    Returns:
        str: Путь к билду среды Unity (например, .app, .x86_64 или .exe).
    """
    
    env_path = os.path.abspath(env_path)
    tmp_dir_path = os.path.join(os.path.dirname(env_path), tmp_dir)

    if zipfile.is_zipfile(env_path):
        logging.info(f"Extracting environment from '{env_path}' to '{tmp_dir_path}'...")
        
        os.makedirs(tmp_dir_path, exist_ok=True)

        try:
            with zipfile.ZipFile(env_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir_path)

            extracted_env_path = None
            for root, dirs, files in os.walk(tmp_dir_path):
                for file in files:
                    if file.endswith((".x86_64", ".exe")):
                        extracted_env_path = os.path.join(root, file)
                        break
                for dir in dirs:
                    if dir.endswith(".app"):
                        extracted_env_path = os.path.join(root, dir)
                        break
                if extracted_env_path:
                    break

            if not extracted_env_path:
                raise FileNotFoundError("No Unity build found in the extracted archive.")

            logging.info(f"Setting permissions 777 to '{tmp_dir_path}'...")
            for root, dirs, files in os.walk(tmp_dir_path):
                for dir in dirs:
                    os.chmod(os.path.join(root, dir), 0o777)
                for file in files:
                    os.chmod(os.path.join(root, file), 0o777)
                    
            logging.info(f"Environment extracted to '{extracted_env_path}'")
            return extracted_env_path

        except Exception as e:
            logging.error(f"Error extracting archive: {e}")
            raise

    else:
        logging.info(f"Using environment at '{env_path}'")
        return env_path

if __name__ == '__main__':
    if not os.path.exists("env.zip"):
        with zipfile.ZipFile("env.zip", "w") as zf:
            zf.writestr("dummy_env.x86_64", "This is a dummy environment file.")

    logging.basicConfig(level=logging.INFO)
    try:
        env_build_path = load_env("env.zip", "tmp_env")
        print(f"Path to Unity build: {env_build_path}")
    except Exception as e:
        print(f"Error: {e}")
