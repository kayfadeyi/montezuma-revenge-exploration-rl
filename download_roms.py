from autorom.rom import download_roms

def main():
    try:
        print("Downloading ROMs...")
        download_roms('.', "accept")
        print("ROMs downloaded successfully!")
    except Exception as e:
        print(f"Error downloading ROMs: {e}")

if __name__ == "__main__":
    main()
