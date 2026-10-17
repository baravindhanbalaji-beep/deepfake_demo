import cv2
print("OpenCV version:", cv2.__version__)
print("Has CascadeClassifier:", hasattr(cv2, 'CascadeClassifier'))
print("Has cv2.data:", hasattr(cv2, 'data'))
if hasattr(cv2, 'legacy'):
    import inspect
    attrs = [a for a in dir(cv2.legacy) if 'cascade' in a.lower() or 'Cascade' in a]
    print("Legacy cascade attrs:", attrs)
else:
    print("No cv2.legacy module")
