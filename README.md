# Sudoku_digits_ocr

Rozpoznawanie cyfr z planszy sudoku

Założenia projektu:
- Program ma być niejako skanerem do sudoku
- Na podstawie zdjęcia ma wyodrębnić z niego plansze sudoku i rozponać zawierające się na niej cyfry
- Wynikiem końcowym projektu jest cyfrowa wersja planszy sudoku

Do nauki modelu wykorzystano zbiór danych MNIST
- Zbior obrazków z ręcznie zapisanymi cyframi
- Obrazy mają rozmiar 28x28 pikseli
- Każdy piksel zapisany jest w skali szarości od 0 do 255
- Zbiór uczący: 60,000 obrazów
- Zbiór testujący: 10,000 obrazów

Schemat działania programu:
- Załadowanie zdjęcia
- Lokalizacja planszy sudoku na zdjęciu
- Lokalizacja pól planszy sudoku
- Ekstrakcja wartości (cyfry lub braku cyfry) z pól
- Rozpoznanie cyfry w danym polu na podstawie wytrenowanego wcześniej modelu sieci
- Zapisanie cyfrowej wersji planszy
- Naniesienie rozpoznanych cyfr na planszę ze zdjęcia

Przyklad dzialania aplikacji:

Dane wejsciowe:
![image](https://github.com/NatanSwierczynski/Sudoku_Digits_Recognition_OCR/assets/106707211/5c7f3e8f-1fb6-4bd0-9154-fc1c382a7d45)

Wyjscie aplikacji:
![image](https://github.com/NatanSwierczynski/Sudoku_Digits_Recognition_OCR/assets/106707211/19ce3eaa-01f8-4adf-b437-2c872d636c4c)




