def set_dpi_awareness():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except ImportError:
        print("ctypes 모듈을 가져오는 데 실패했습니다. Python 환경을 확인하세요.")
    except AttributeError:
        print("SetProcessDpiAwareness 함수가 지원되지 않는 환경입니다. Windows 8.1 이상에서만 지원됩니다.")
    except OSError as os_error:
        print(f"OS 관련 오류 발생: {os_error}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")