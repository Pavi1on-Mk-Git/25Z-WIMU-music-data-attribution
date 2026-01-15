# WIMU/SIWY - Projekt - Obecne postępy

## Test na zoverfitowanym modelu - opis
- zbiór danych - 64 losowe próbki ze zbioru treningowego
- model - zoverfitowany na ww. zbiorze danych
- używamy zoverfitowanego modelu, by wygenerować próbki używając promptów odpowiadających próbkom z ww. zbioru danych
- uruchamiamy TRAK na tym modelu i wygenerowanych próbkach
- mierzymy accuracy próbki odpowiadającej jej wygenerowanej wersji znajdującej się w top-1 i top-5 najwyższych scorów

## Postępy

### MusicGen

- powstały skrypty pozwalające na wygenerowanie checkpointów potrzebnych do zaaplikowania metody TRAK

- udało się uruchomić test na zoverfitowanym modelu - wynik niezadowalający niezależnie od konfiguracji (sprawdzone trzy warianty model output function, każda z wykorzystaniem CFG do wyliczania tokenów i bez)

- udało się uruchomić TRAK na 100 docelowych checkpointach uczonych na 0.5 całego zbioru treningowego każdy - wynik niezadowalający

### SAO

- powstał skrypt pozwalający na wygenerowanie checkpointów potrzebnych do zaaplikowania metody TRAK

- udało się uruchomić test na zoverfitowanym modelu - wynik obiecujący (top1: 0.69, top5: 0.84)
