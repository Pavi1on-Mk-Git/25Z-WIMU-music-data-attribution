# WIMU/SIWY - Projekt - Design proposal
Jakub Proboszcz, Paweł Kochański, Mikołaj Piórczyński
## Atrybucja danych audio wykorzystując TRAK dla dyfuzyjnych modeli generatywnych audio

### Planowany zakres projektu

Celem projektu jest uruchomienie i zbadanie metody TRAK [1] dla modeli generujących audio, np. Stable Audio Open [2]. Na potrzeby metody TRAK modele będą wielokrotnie dotrenowywane; jako zbiór danych wstępnie proponujemy zbiór AudioCaps [3].
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
- oficjalna implementacja metody [TRAK](https://github.com/MadryLab/trak), ewentualnie [D-TRAK](https://github.com/sail-sg/D-TRAK) lub [journey-TRAK](https://github.com/MadryLab/journey-TRAK/tree/main),
- dotrenowywanie modeli: pytorch, torchaudio, huggingface, ewentualnie też biblioteki poszczególnych modeli

Narzędzia pomocnicze:
- autoformatter: ruff
- linter: flake8
- zarządzanie projektem: just
- środowisko wirtualne i zarządzanie zależnościami: pdm
- testy: pytest, ew. skrypty shellowe

### Wymagane zasoby obliczeniowe

Założenia:
- model Stable Audio Open [2]
- karta graficzna NVIDIA A100 [4]
- zbiór danych AudioCaps [3]
- `batch_size` równy $4$
- do treningu używana będzie połowa ww. zbioru

Czas trwania pojedynczej epoki na jednym GPU $\approx 8.86 \space h$

### Prace powiązane z tematem

[5] i [6] są przykładami wykorzystania metody TRAK dla modeli dyfuzyjnych generujących obrazy.

[7] jest przykładem wykorzystania metody TRAK na modelu autoregresyjnym generującym muzykę symboliczną w formacie MIDI.

### Bibliografia

[1] Sung Min Park i in., "TRAK: Attributing Model Behavior at Scale" w International Conference on Machine Learning (ICML), 2023, https://arxiv.org/pdf/2303.14186.

[2] Zach Evans i in., "Stable Audio Open", arXiv preprint, 2024, https://arxiv.org/pdf/2407.14358.

[3] Chris Dongjoo Kim i in., "AudioCaps: Generating Captions for Audios in The Wild" w Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, https://doi.org/10.18653/v1/N19-1011.

[4] NVIDIA, "NVIDIA A100 | NVIDIA", 2024, https://www.nvidia.com/en-us/data-center/a100/, ostatni dostęp 03.11.2025.

[5] Zheng Xiaosen i in., "Intriguing Properties of Data Attribution on Diffusion Models" w International Conference on Learning Representations (ICLR), 2024, https://arxiv.org/pdf/2311.00500.

[6] Kristian Georgiev i in., "The Journey, Not the Destination: How Data Guides Diffusion Models", arXiv preprint, 2023, https://arxiv.org/abs/2312.06205.

[7] Junwei Deng, Shiyuan Zhang, Jiaqi Ma, "Computational Copyright: Towards A Royalty Model for Music Generative AI", arXiv preprint, 2023, https://arxiv.org/abs/2312.06646.

