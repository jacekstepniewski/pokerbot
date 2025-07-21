import cv2
import os
from glob import glob

def preprocess_templates(input_dir, output_dir,
                         convert_gray=True,
                         equalize_hist=True,
                         gaussian_blur=True,
                         adaptive_thresh=False,
                         canny_edges=False,
                         blur_kernel=(5,5),
                         canny_thresholds=(50,150),
                         resize_to=None):
    """
    Ładuje wszystkie obrazki z input_dir, przeprowadza zestaw operacji przetwarzania
    i zapisuje wynik w output_dir z tymi samymi nazwami plików.
    """
    os.makedirs(output_dir, exist_ok=True)
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    for pattern in patterns:
        for path in glob(os.path.join(input_dir, pattern)):
            img = cv2.imread(path)
            proc = img.copy()

            # 1. Konwersja na skalę szarości
            if convert_gray:
                proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            
            # 2. Wyrównanie histogramu
            if equalize_hist:
                proc = cv2.equalizeHist(proc)

            # 3. Rozmycie Gaussa
            if gaussian_blur:
                proc = cv2.GaussianBlur(proc, blur_kernel, 0)

            # 4. Progowanie adaptacyjne
            if adaptive_thresh:
                proc = cv2.adaptiveThreshold(proc, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

            # 5. Detekcja krawędzi Canny'ego
            if canny_edges:
                proc = cv2.Canny(proc, *canny_thresholds)

            # 6. Zmiana rozmiaru
            if resize_to is not None:
                proc = cv2.resize(proc, resize_to, interpolation=cv2.INTER_AREA)

            # Zapis wyniku
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, proc)
            print(f"Zapisano: {save_path}")

if __name__ == "__main__":
    # Przykład użycia:
    preprocess_templates(
        input_dir="pokerbot/rank/",
        output_dir="templates_processed/",
        convert_gray=True,
        equalize_hist=True,
        gaussian_blur=True,
        adaptive_thresh=False,
        canny_edges=False,
        resize_to=None
    )

