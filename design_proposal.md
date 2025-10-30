# WIMU/SIWY - Projekt - Design proposal
Jakub Proboszcz, Paweł Kochański, Kamil Michalak
## Atrybucja danych audio wykorzystując TRAK dla dyfuzyjnych modeli generatywnych audio

### Planowany zakres projektu

Celem projektu jest uruchomienie i zbadanie metody [TRAK](https://arxiv.org/pdf/2510.08062) dla modeli generujących audio, np. [Stable Audio Open](https://arxiv.org/pdf/2407.14358).
W ramach projektu powstanie artykuł gotowy lub bliski gotowości do zgłoszenia na konferencję naukową.

### Harmonogram projektu

| Tydzień         | Zaplanowane efekty  |
|-----------------|---------------------|
| 3 XI - 9 XI     | Uruchomienie istniejącej implementacji metody TRAK. |
| 10 XI - 16 XI   | Zidentyfikowanie i uruchomienie dostępnych do pobrania modeli spełniających warunki tematu. |
| 17 XI - 23 XI   | Przygotowanie dotrenowywania modeli. |
| 24 XI - 30 XI   | Przygotowanie dotrenowywania modeli. |
| 1 XII - 7 XII   | Trening modeli. |
| 8 XII - 14 XII  | Trening modeli. |
| 15 XII - 21 XII | Trening modeli. |
| 22 XII - 28 XII | - |
| 29 XII - 4 I    | Uruchomienie metody TRAK z wykorzystaniem checkpointów modeli. |
| 5 I - 11 I      | Przygotowanie artykułu. |
| 12 I - 18 I     | Ewentualne poprawki. |
| 19 I - 25 I     | - |
| 26 I            | Ostateczny termin oddania projektu. |

### Planowane technologie

Język Python oraz biblioteki:
- oficjalna implementacja metody TRAK: [link](https://github.com/MadryLab/trak)
- dotrenowywanie modeli: pytorch, torchaudio, ewentualnie też biblioteki poszczególnych modeli

Narzędzia pomocnicze:
- autoformatter: black
- linter: flake8
- zarządzanie projektem: just
- środowisko wirtualne i zarządzanie zależnościami: pdm
- testy: pytest, ew. skrypty shellowe

### Wymagane zasoby obliczeniowe
